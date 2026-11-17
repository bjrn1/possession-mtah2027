import numpy as np

from possession.ball import remove_spikes
from possession.codes import runs
from possession.features import FPS, home_attacks_right
from possession.io_atd import HOME, Tracking

CAP_S = 60.0
NEAR_M = 1.5
REL_MS = 2.0
CONTROL_FRAMES = 10
TURN_DEG = 30.0
TURN_SPEED_MS = 3.0
CONTACT_MS = 3.0
DEDUP_FRAMES = 8
VEL_HALF = 2
NAMES = (
    "ball_since_seen",
    "ball_until_seen",
    "ball_gap_len",
    "ball_last_line",
    "ball_next_line",
    "ball_last_speed",
    "ball_next_speed",
    "ball_last_x",
    "ball_next_x",
    "ball_last_near_diff",
    "ball_next_near_diff",
    "control_now",
    "control_last_team",
    "control_since",
    "control_next_team",
    "control_until",
    "touch_last_team",
    "touch_since",
    "touch_next_team",
    "touch_until",
    "deflection_since",
    "deflection_until",
    "control_soft_c12",
    "control_soft_c50",
    "control_soft_c125",
)


def _velocity(xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape, np.nan, dtype=np.float64)
    k = VEL_HALF
    out[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    return out


def _last_next(frames: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(n)
    if frames.size == 0:
        return np.zeros(n), np.full(n, CAP_S), np.zeros(n), np.full(n, CAP_S)
    left = np.searchsorted(frames, index, side="right") - 1
    has_left = left >= 0
    safe_left = np.maximum(left, 0)
    last = np.where(has_left, values[safe_left], 0.0)
    since = np.where(has_left, (index - frames[safe_left]) / FPS, CAP_S)
    right = np.searchsorted(frames, index, side="left")
    has_right = right < frames.size
    safe_right = np.minimum(right, frames.size - 1)
    following = np.where(has_right, values[safe_right], 0.0)
    until = np.where(has_right, (frames[safe_right] - index) / FPS, CAP_S)
    return last, np.clip(since, 0.0, CAP_S), following, np.clip(until, 0.0, CAP_S)


def _seen_last_next(seen: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.flatnonzero(seen)
    n = seen.size
    if frames.size == 0:
        return np.full(n, np.nan), np.full(n, np.nan)
    index = np.arange(n)
    left = np.searchsorted(frames, index, side="right") - 1
    right = np.searchsorted(frames, index, side="left")
    last = np.where(left >= 0, values[frames[np.maximum(left, 0)]], np.nan)
    following = np.where(right < frames.size, values[frames[np.minimum(right, frames.size - 1)]], np.nan)
    return last, following


def _centred_mean(values: np.ndarray, half: int) -> np.ndarray:
    n = values.size
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    index = np.arange(n)
    lo = np.clip(index - half, 0, n)
    hi = np.clip(index + half + 1, 0, n)
    return (cumulative[hi] - cumulative[lo]) / np.maximum(hi - lo, 1)


def _episodes(team: np.ndarray, minimum: int) -> list[tuple[int, int, int]]:
    out = []
    start = 0
    n = team.size
    for t in range(1, n + 1):
        if t == n or team[t] != team[start]:
            if team[start] != 0 and t - start >= minimum:
                out.append((start, t, int(team[start])))
            start = t
    return out


def _block(tracking: Tracking, start: int, stop: int) -> np.ndarray:
    length, width = tracking.pitch
    n = stop - start
    xy = tracking.players[start:stop, :, :2].astype(np.float64)
    raw_ball = tracking.ball[start:stop].astype(np.float64).copy()
    if not home_attacks_right(tracking, start, stop):
        xy[..., 0] = length - xy[..., 0]
        xy[..., 1] = width - xy[..., 1]
        raw_ball[:, 0] = length - raw_ball[:, 0]
        raw_ball[:, 1] = width - raw_ball[:, 1]
    ball = remove_spikes(raw_ball, np.ones(n, dtype=bool))
    seen = ~np.isnan(ball[:, 0])
    sign_of_slot = np.where(tracking.meta.slot_team == HOME, 1.0, -1.0)

    ball_velocity = _velocity(ball)
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    player_velocity = _velocity(xy)
    distance = np.linalg.norm(xy - ball[:, None, :], axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    relative = np.linalg.norm(player_velocity - ball_velocity[:, None, :], axis=2)
    relative = np.where(np.isnan(relative), np.inf, relative)

    home_mask = tracking.meta.slot_team == HOME
    near_home = np.where(home_mask[None, :], distance, np.inf).min(axis=1)
    near_away = np.where(~home_mask[None, :], distance, np.inf).min(axis=1)
    with np.errstate(invalid="ignore"):
        near_diff = np.clip(np.where(np.isfinite(near_home) & np.isfinite(near_away), near_home - near_away, np.nan), -30.0, 30.0)

    nearest = np.argmin(distance, axis=1)
    rows = np.arange(n)
    nearest_distance = distance[rows, nearest]
    nearest_sign = np.where(np.isfinite(nearest_distance), sign_of_slot[nearest], 0.0)

    with_ball = (distance < NEAR_M) & (relative < REL_MS)
    controller = np.argmin(np.where(with_ball, distance, np.inf), axis=1)
    has_controller = with_ball[rows, controller]
    control_now = np.where(has_controller, sign_of_slot[controller], 0.0)

    control_episodes = _episodes(control_now.astype(np.int8), CONTROL_FRAMES)
    controlled = np.zeros(n, dtype=bool)
    for a, b, _ in control_episodes:
        controlled[a:b] = True

    k = 3
    before = ball_velocity[:-2 * k]
    after = ball_velocity[2 * k:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(before * after, axis=1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1))
    turn = np.zeros(n, dtype=bool)
    angle = np.degrees(np.arccos(np.clip(np.nan_to_num(cosine, nan=1.0), -1.0, 1.0)))
    turn[k:-k] = (angle > TURN_DEG) & (np.nan_to_num(np.linalg.norm(before, axis=1)) > TURN_SPEED_MS) & (np.nan_to_num(np.linalg.norm(after, axis=1)) > TURN_SPEED_MS)
    close = nearest_distance < NEAR_M
    deflections = []
    last = -10 ** 9
    for t in np.flatnonzero(turn & close):
        if t - last < DEDUP_FRAMES:
            continue
        team = nearest_sign[t]
        follow = slice(t, min(n, t + CONTROL_FRAMES + 1))
        if np.any((control_now[follow] == team) & controlled[follow]):
            continue
        deflections.append((int(t), float(team)))
        last = t
    change = np.zeros(n)
    change[k:-k] = np.nan_to_num(np.linalg.norm(ball_velocity[2 * k:] - ball_velocity[:-2 * k], axis=1))
    contacts = []
    last_by_team = {1.0: -10 ** 9, -1.0: -10 ** 9}
    for t in np.flatnonzero((change > CONTACT_MS) & close):
        team = float(nearest_sign[t])
        if team == 0.0 or t - last_by_team[team] < DEDUP_FRAMES:
            continue
        last_by_team[team] = t
        contacts.append((int(t), team))

    features = {}
    _, features["ball_since_seen"], _, features["ball_until_seen"] = _last_next(np.flatnonzero(seen), np.zeros(int(seen.sum())), n)
    features["ball_gap_len"] = np.minimum(features["ball_since_seen"] + features["ball_until_seen"], 2 * CAP_S)
    line = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    features["ball_last_line"], features["ball_next_line"] = _seen_last_next(seen, line)
    speed_seen = seen & np.isfinite(ball_speed)
    features["ball_last_speed"], features["ball_next_speed"] = _seen_last_next(speed_seen, np.nan_to_num(ball_speed))
    features["ball_last_x"], features["ball_next_x"] = _seen_last_next(seen, ball[:, 0])
    near_seen = seen & np.isfinite(near_diff)
    features["ball_last_near_diff"], features["ball_next_near_diff"] = _seen_last_next(near_seen, np.nan_to_num(near_diff))
    features["control_now"] = control_now

    frames = np.array([a for a, _, _ in control_episodes], dtype=np.int64)
    teams = np.array([float(s) for _, _, s in control_episodes])
    (features["control_last_team"], features["control_since"], features["control_next_team"], features["control_until"]) = _last_next(frames, teams, n)

    touches = sorted([(a, float(s)) for a, _, s in control_episodes] + deflections + contacts)
    frames = np.array([t for t, _ in touches], dtype=np.int64)
    teams = np.array([s for _, s in touches])
    (features["touch_last_team"], features["touch_since"], features["touch_next_team"], features["touch_until"]) = _last_next(frames, teams, n)

    frames = np.array([t for t, _ in deflections], dtype=np.int64)
    _, features["deflection_since"], _, features["deflection_until"] = _last_next(frames, np.ones(frames.size), n)

    with np.errstate(invalid="ignore"):
        gap = np.where(np.isfinite(near_home) & np.isfinite(near_away), (near_away - near_home) / 0.25, 0.0)
    soft = np.where(np.minimum(near_home, near_away) < NEAR_M, np.clip(gap, -1.0, 1.0), 0.0)
    for half_width, name in ((12, "control_soft_c12"), (50, "control_soft_c50"), (125, "control_soft_c125")):
        features[name] = _centred_mean(soft, half_width)

    return np.stack([np.asarray(features[name], dtype=np.float32) for name in NAMES], axis=1)


def control_features(tracking: Tracking) -> np.ndarray:
    values = np.full((tracking.n_frames, len(NAMES)), np.nan, dtype=np.float32)
    for start, stop in runs(tracking.present):
        values[start:stop] = _block(tracking, start, stop)
    return values
