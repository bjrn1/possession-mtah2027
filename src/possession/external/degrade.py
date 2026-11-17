from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from possession.codes import runs
from possession.external.dfl import ExternalMatch
from possession.io_atd import AWAY, HOME, Tracking, TrackingMeta

FPS = 25
BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16
EDGE_MARGIN = 3.0
BALL_CROWD_RADIUS = 2.0
PLAYER_CROWD_RADIUS = 1.5
SWAP_RADIUS = 1.0
CHUNK = 8192


@dataclass(frozen=True)
class DegradeRanges:
    ball_missing: tuple[float, float] = (0.25, 0.55)
    ball_gap_seconds: tuple[float, float] = (0.4, 2.5)
    ball_crowd_weight: tuple[float, float] = (0.3, 0.8)
    ball_height_weight: tuple[float, float] = (0.5, 1.5)
    ball_speed_weight: tuple[float, float] = (0.2, 0.8)
    ball_box_weight: tuple[float, float] = (0.3, 1.0)
    ball_edge_weight: tuple[float, float] = (0.0, 0.6)
    ball_noise_m: tuple[float, float] = (0.1, 0.5)
    ball_false_rate_per_minute: tuple[float, float] = (0.0, 0.5)
    player_missing: tuple[float, float] = (0.03, 0.12)
    player_gap_seconds: tuple[float, float] = (0.3, 2.0)
    player_crowd_weight: tuple[float, float] = (0.3, 1.0)
    player_edge_weight: tuple[float, float] = (0.2, 1.0)
    fragment_seconds: tuple[float, float] = (8.0, 60.0)
    swap_probability: tuple[float, float] = (0.05, 0.3)
    duplicate_rate_per_minute: tuple[float, float] = (0.0, 0.4)
    duplicate_seconds: tuple[float, float] = (0.5, 3.0)
    player_noise_m: tuple[float, float] = (0.05, 0.3)
    slot_reuse_gap_seconds: tuple[float, float] = (0.5, 5.0)


@dataclass(frozen=True)
class DegradeParams:
    seed: int
    ball_missing: float
    ball_gap_seconds: float
    ball_crowd_weight: float
    ball_height_weight: float
    ball_speed_weight: float
    ball_box_weight: float
    ball_edge_weight: float
    ball_noise_m: float
    ball_false_rate_per_minute: float
    player_missing: float
    player_gap_seconds: float
    player_crowd_weight: float
    player_edge_weight: float
    fragment_seconds: float
    swap_probability: float
    duplicate_rate_per_minute: float
    duplicate_seconds: float
    player_noise_m: float
    slot_reuse_gap_seconds: float


def sample_params(seed: int, ranges: DegradeRanges = DegradeRanges()) -> DegradeParams:
    rng = np.random.default_rng([seed, 0])
    values = {name: float(rng.uniform(*bounds)) for name, bounds in asdict(ranges).items()}
    return DegradeParams(seed=seed, **values)


def _in_box(xy: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    length, width = pitch
    near_goal = (xy[..., 0] < BOX_DEPTH) | (xy[..., 0] > length - BOX_DEPTH)
    central = np.abs(xy[..., 1] - width / 2.0) < BOX_HALF_WIDTH
    return near_goal & central


def _near_edge(xy: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    length, width = pitch
    x, y = xy[..., 0], xy[..., 1]
    return (x < EDGE_MARGIN) | (x > length - EDGE_MARGIN) | (y < EDGE_MARGIN) | (y > width - EDGE_MARGIN)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, ...], sigma: float, window: int) -> np.ndarray:
    white = rng.normal(0.0, 1.0, (shape[0] + window,) + shape[1:])
    cumulative = np.cumsum(white, axis=0)
    smooth = (cumulative[window:] - cumulative[:-window]) / np.sqrt(window)
    return (smooth * sigma).astype(np.float32)


def _speed_from_positions(xy: np.ndarray) -> np.ndarray:
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    delta = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) * FPS / 4.0
    speed[2:-2] = delta
    return speed


def _ball_speed(ball: np.ndarray) -> np.ndarray:
    step = np.zeros(ball.shape[0], dtype=np.float32)
    delta = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    step[1:] = np.nan_to_num(delta, nan=0.0)
    return np.minimum(step, 40.0)


def _ball_crowd(players: np.ndarray, ball: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(players[:, :, :2] - ball[:, None, :], axis=2)
    return np.sum(distance < BALL_CROWD_RADIUS, axis=1).astype(np.float32)


def _player_crowd(players: np.ndarray) -> np.ndarray:
    n_frames, n_players = players.shape[:2]
    out = np.zeros((n_frames, n_players), dtype=np.float32)
    for start in range(0, n_frames, CHUNK):
        xy = players[start:start + CHUNK, :, :2]
        distance = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=3)
        out[start:start + CHUNK] = np.sum(distance < PLAYER_CROWD_RADIUS, axis=2) - 1
    return np.maximum(out, 0.0)


def _gap_mask(
    hazard: np.ndarray,
    valid: np.ndarray,
    uniforms: np.ndarray,
    lengths: np.ndarray,
    block_end: np.ndarray,
) -> np.ndarray:
    size = hazard.shape[0]
    starts = np.flatnonzero(valid & (uniforms < hazard))
    stops = np.minimum(np.minimum(starts + lengths[starts], block_end[starts]), size)
    edges = np.zeros(size + 1, dtype=np.int32)
    np.add.at(edges, starts, 1)
    np.add.at(edges, stops, -1)
    return (np.cumsum(edges[:-1]) > 0) & valid


def _calibrated_mask(
    logit_without_base: np.ndarray,
    valid: np.ndarray,
    target: float,
    uniforms: np.ndarray,
    lengths: np.ndarray,
    block_end: np.ndarray,
) -> np.ndarray:
    if not valid.any():
        return np.zeros_like(valid)
    low, high = -14.0, 4.0
    mask = np.zeros_like(valid)
    for _ in range(30):
        middle = (low + high) / 2.0
        hazard = 1.0 / (1.0 + np.exp(-(middle + logit_without_base)))
        mask = _gap_mask(hazard, valid, uniforms, lengths, block_end)
        share = mask[valid].mean()
        if share < target:
            low = middle
        else:
            high = middle
    return mask


def _block_end(half: np.ndarray) -> np.ndarray:
    out = np.zeros(half.size, dtype=np.int64)
    for start, stop in runs(half > 0):
        out[start:stop] = stop
    return out


def _degrade_ball(match: ExternalMatch, params: DegradeParams, rng: np.random.Generator) -> np.ndarray:
    ball = match.ball.copy()
    valid = (match.half > 0) & ~np.isnan(ball[:, 0])
    crowd = _ball_crowd(match.players, ball)
    height = np.clip(np.nan_to_num(match.ball_z, nan=0.0) / 2.0, 0.0, 2.0)
    speed = _ball_speed(ball) / 20.0
    box = _in_box(ball, match.pitch).astype(np.float32)
    edge = _near_edge(ball, match.pitch).astype(np.float32)
    logit = (
        params.ball_crowd_weight * crowd
        + params.ball_height_weight * height
        + params.ball_speed_weight * speed
        + params.ball_box_weight * box
        + params.ball_edge_weight * edge
    )
    uniforms = rng.random(ball.shape[0])
    lengths = rng.geometric(1.0 / max(1.0, params.ball_gap_seconds * FPS), ball.shape[0])
    lost = _calibrated_mask(logit, valid, params.ball_missing, uniforms, lengths, _block_end(match.half))

    ball += _smooth_noise(rng, ball.shape, params.ball_noise_m, 5)
    ball[lost] = np.nan

    played = np.flatnonzero(match.half > 0)
    n_false = rng.poisson(params.ball_false_rate_per_minute * played.size / (60.0 * FPS))
    for start in rng.choice(played, size=n_false, replace=False) if n_false else []:
        duration = int(rng.integers(1, 6))
        stop = min(ball.shape[0], start + duration)
        ball[start:stop] = rng.uniform([0.0, 0.0], match.pitch, size=(stop - start, 2))
    ball[match.half == 0] = np.nan
    return ball


def _degrade_presence(match: ExternalMatch, params: DegradeParams, rng: np.random.Generator) -> np.ndarray:
    players = match.players
    seen = ~np.isnan(players[:, :, 0]) & (match.half > 0)[:, None]
    crowd = _player_crowd(players)
    edge = _near_edge(players[:, :, :2], match.pitch).astype(np.float32)
    logit = params.player_crowd_weight * crowd + params.player_edge_weight * edge
    uniforms = rng.random(seen.shape)
    lengths = rng.geometric(1.0 / max(1.0, params.player_gap_seconds * FPS), seen.shape)
    n_frames, n_people = seen.shape
    offsets = (np.arange(n_people) * n_frames)[:, None]
    block_end = (_block_end(match.half)[None, :] + offsets).ravel()
    flat_lost = _calibrated_mask(
        logit.T.ravel(), seen.T.ravel(), params.player_missing, uniforms.T.ravel(), lengths.T.ravel(), block_end
    )
    lost = flat_lost.reshape(seen.shape[1], seen.shape[0]).T
    return seen & ~lost


def _swap_events(match: ExternalMatch, visible: np.ndarray, params: DegradeParams, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    events = []
    xy = match.players[:, :, :2]
    for team in (HOME, AWAY):
        members = np.flatnonzero(match.slot_team == team)
        for a_index, a in enumerate(members):
            for b in members[a_index + 1:]:
                close = visible[:, a] & visible[:, b] & (np.linalg.norm(xy[:, a] - xy[:, b], axis=1) < SWAP_RADIUS)
                for start, _ in runs(close):
                    if start > 0 and rng.random() < params.swap_probability:
                        events.append((start, int(a), int(b)))
    return sorted(events)


def _fragments(visible: np.ndarray, swaps: list[tuple[int, int, int]], params: DegradeParams, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    cuts = rng.random(visible.shape) < 1.0 / max(1.0, params.fragment_seconds * FPS)
    for frame, a, b in swaps:
        cuts[frame, a] = True
        cuts[frame, b] = True
    out = []
    for person in range(visible.shape[1]):
        for start, stop in runs(visible[:, person]):
            bounds = [start] + [int(c) for c in np.flatnonzero(cuts[start + 1:stop, person]) + start + 1] + [stop]
            out.extend((s, e, person) for s, e in zip(bounds[:-1], bounds[1:]) if e > s)
    return sorted(out)


def _allocate_slots(
    fragments: list[tuple[int, int, int]],
    person_team: np.ndarray,
    swaps: list[tuple[int, int, int]],
    params: DegradeParams,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray]:
    reuse_gap = int(params.slot_reuse_gap_seconds * FPS)
    partner = {}
    for frame, a, b in swaps:
        partner[(frame, a)] = b
        partner[(frame, b)] = a
    slot_team: list[int] = []
    slot_free_at: list[int] = []
    slot_of_end: dict[tuple[int, int], int] = {}
    assignment = []
    for start, stop, person in fragments:
        team = int(person_team[person])
        inherited = None
        if (start, person) in partner:
            inherited = slot_of_end.pop((start, partner[(start, person)]), None)
        if inherited is not None and slot_free_at[inherited] <= start:
            slot = inherited
        else:
            free = [s for s in range(len(slot_team)) if slot_team[s] == team and slot_free_at[s] + reuse_gap <= start]
            if free:
                slot = int(free[int(rng.integers(len(free)))])
            else:
                slot = len(slot_team)
                slot_team.append(team)
                slot_free_at.append(0)
        slot_free_at[slot] = stop
        slot_of_end[(stop, person)] = slot
        assignment.append(slot)
    return assignment, np.array(slot_team, dtype=np.int8)


def degrade(match: ExternalMatch, params: DegradeParams) -> Tracking:
    rng = np.random.default_rng([params.seed, 1])
    ball = _degrade_ball(match, params, rng)
    visible = _degrade_presence(match, params, rng)
    swaps = _swap_events(match, visible, params, rng)
    fragments = _fragments(visible, swaps, params, rng)
    assignment, slot_team = _allocate_slots(fragments, match.slot_team, swaps, params, rng)

    n_frames = match.n_frames
    source = match.players.copy()
    source[:, :, :2] += _smooth_noise(rng, (n_frames, match.players.shape[1], 2), params.player_noise_m, FPS)
    slots = np.full((n_frames, slot_team.size, 3), np.nan, dtype=np.float32)
    for (start, stop, person), slot in zip(fragments, assignment):
        slots[start:stop, slot] = source[start:stop, person]

    played = np.flatnonzero(match.half > 0)
    n_duplicates = rng.poisson(params.duplicate_rate_per_minute * played.size / (60.0 * FPS))
    extra = []
    for _ in range(n_duplicates):
        start = int(rng.choice(played))
        stop = min(n_frames, start + int(params.duplicate_seconds * FPS))
        candidates = np.flatnonzero(visible[start])
        if candidates.size == 0:
            continue
        person = int(rng.choice(candidates))
        column = np.full((n_frames, 3), np.nan, dtype=np.float32)
        column[start:stop] = source[start:stop, person]
        column[start:stop, :2] += _smooth_noise(rng, (stop - start, 2), 0.5, FPS)
        column[start:stop][~visible[start:stop, person]] = np.nan
        extra.append((column, int(match.slot_team[person])))
    if extra:
        slots = np.concatenate([slots, np.stack([c for c, _ in extra], axis=1)], axis=1)
        slot_team = np.concatenate([slot_team, np.array([t for _, t in extra], dtype=np.int8)])

    order = rng.permutation(slot_team.size)
    slots = slots[:, order]
    slot_team = slot_team[order]
    slots[:, :, 2] = _speed_from_positions(slots[:, :, :2])

    meta = TrackingMeta(team_ids=("home", "away"), slot_team=slot_team)
    return Tracking(meta=meta, players=slots, ball=ball, present=match.half > 0, pitch=match.pitch)


def save_tracking(tracking: Tracking, params: DegradeParams, path: str | Path) -> None:
    np.savez_compressed(
        path,
        players=tracking.players,
        ball=tracking.ball,
        present=tracking.present,
        slot_team=tracking.meta.slot_team,
        pitch=np.array(tracking.pitch),
        params=np.array([str(asdict(params))]),
    )


def load_tracking(path: str | Path) -> Tracking:
    with np.load(path) as data:
        meta = TrackingMeta(team_ids=("home", "away"), slot_team=data["slot_team"])
        return Tracking(
            meta=meta,
            players=data["players"],
            ball=data["ball"],
            present=data["present"],
            pitch=tuple(float(v) for v in data["pitch"]),
        )
