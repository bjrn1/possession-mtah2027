from dataclasses import dataclass

import numpy as np

from possession.ball import linear_fill, remove_spikes
from possession.codes import runs
from possession.io_atd import AWAY, HOME, Tracking, TrackingMeta

FPS = 25
FAR = 60.0
GAP_CAP = 250
SLOW_SPEED = 1.5
POSSESSION_RADIUS = 1.5
DUEL_RADIUS = 3.0
CENTRED_WINDOWS = (5, 25, 75, 250)
SIDED_WINDOWS = (25, 75)
CONTEXT_FEATURES = (
    "ball_seen",
    "ball_speed",
    "ball_accel",
    "ball_turn",
    "ball_edge_distance",
    "ball_inside",
    "all_speed_mean",
    "all_slow_share",
    "all_speed_std",
    "near_h",
    "near_a",
    "near_diff",
    "count_poss_h",
    "count_poss_a",
    "rel_speed_h",
    "rel_speed_a",
    "centroid_vx_h",
    "centroid_vx_a",
    "spread_x_diff",
    "spread_y_diff",
    "centroid_x_diff",
    "players_outside",
    "corner_distance",
)


@dataclass(frozen=True)
class FrameFeatures:
    values: np.ndarray
    names: tuple[str, ...]


def _central_speed(xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    out[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) * FPS / 4.0
    return out


def _central_velocity(xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    out[2:-2] = (xy[4:] - xy[:-4]) * FPS / 4.0
    return out


def distance_to_observation(seen: np.ndarray) -> np.ndarray:
    n = seen.size
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    back = np.where(last >= 0, index - last, GAP_CAP)
    forward = np.where(following < n, following - index, GAP_CAP)
    return np.minimum(np.minimum(back, forward), GAP_CAP)


def home_attacks_right(tracking: Tracking, start: int, stop: int) -> bool:
    x = tracking.players[start:stop, :, 0]
    home = x[:, tracking.meta.slot_team == HOME]
    away = x[:, tracking.meta.slot_team == AWAY]
    if not np.isfinite(home).any() or not np.isfinite(away).any():
        return True
    return bool(np.nanmean(home) < np.nanmean(away))


def _nan_stat(values: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    data = np.where(mask, values, np.nan)
    count = np.sum(mask, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.nansum(data, axis=1)
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        if kind == "mean":
            return mean.astype(np.float32)
        square = np.nansum(data * data, axis=1)
        variance = np.where(count > 1, square / np.maximum(count, 1) - mean * mean, np.nan)
        return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def _base_block(tracking: Tracking, start: int, stop: int) -> dict[str, np.ndarray]:
    length, width = tracking.pitch
    players = tracking.players[start:stop].astype(np.float32)
    xy = players[:, :, :2].copy()
    attacks_right = home_attacks_right(tracking, start, stop)
    if not attacks_right:
        xy[..., 0] = length - xy[..., 0]
        xy[..., 1] = width - xy[..., 1]
    seen_player = ~np.isnan(xy[..., 0])
    speed = _central_speed(xy)
    velocity = _central_velocity(xy)
    home = (tracking.meta.slot_team == HOME)[None, :] & seen_player
    away = (tracking.meta.slot_team == AWAY)[None, :] & seen_player
    anyone = seen_player

    raw_ball = tracking.ball[start:stop].astype(np.float64).copy()
    if not attacks_right:
        raw_ball[:, 0] = length - raw_ball[:, 0]
        raw_ball[:, 1] = width - raw_ball[:, 1]
    present = np.ones(stop - start, dtype=bool)
    clean = remove_spikes(raw_ball, present)
    ball_seen = ~np.isnan(clean[:, 0])
    ball = linear_fill(clean, present).astype(np.float32)
    if not ball_seen.any():
        ball[:] = [length / 2.0, width / 2.0]
    ball_velocity = _central_velocity(ball)
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    ball_accel = np.full(ball_speed.shape, np.nan, dtype=np.float32)
    ball_accel[2:-2] = np.linalg.norm(ball_velocity[4:] - ball_velocity[:-4], axis=1) * FPS / 4.0
    before = np.full(ball.shape, np.nan, dtype=np.float32)
    after = np.full(ball.shape, np.nan, dtype=np.float32)
    before[3:] = ball[3:] - ball[:-3]
    after[:-3] = ball[3:] - ball[:-3]
    norms = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(before * after, axis=1) / norms
    ball_turn = np.where(norms > 0.01, 1.0 - np.clip(cosine, -1.0, 1.0), 0.0).astype(np.float32)

    bx, by = ball[:, 0], ball[:, 1]
    edge = np.minimum.reduce([bx, length - bx, by, width - by])
    inside = ((bx >= 0) & (bx <= length) & (by >= 0) & (by <= width)).astype(np.float32)

    distance = np.linalg.norm(xy - ball[:, None, :], axis=2)
    distance = np.where(seen_player, distance, np.inf)

    def nearest(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = np.where(mask, distance, np.inf)
        order = np.argsort(d, axis=1)[:, :2]
        rows = np.arange(d.shape[0])
        first = np.minimum(d[rows, order[:, 0]], FAR)
        second = np.minimum(d[rows, order[:, 1]], FAR)
        player_velocity = velocity[rows, order[:, 0]]
        relative = np.linalg.norm(ball_velocity - player_velocity, axis=1)
        relative = np.where(first < FAR, relative, np.nan)
        return first.astype(np.float32), second.astype(np.float32), relative.astype(np.float32)

    near_h, second_h, rel_h = nearest(home)
    near_a, second_a, rel_a = nearest(away)

    def team_shape(mask: np.ndarray) -> dict[str, np.ndarray]:
        x, y = xy[..., 0], xy[..., 1]
        cx = _nan_stat(x, mask, "mean")
        cy = _nan_stat(y, mask, "mean")
        return {
            "count": np.sum(mask, axis=1).astype(np.float32),
            "cx": cx,
            "cy": cy,
            "sx": _nan_stat(x, mask, "std"),
            "sy": _nan_stat(y, mask, "std"),
            "speed": _nan_stat(speed, mask & ~np.isnan(speed), "mean"),
            "slow": _nan_stat((speed < SLOW_SPEED).astype(np.float32), mask & ~np.isnan(speed), "mean"),
            "own_half": _nan_stat((x < length / 2.0).astype(np.float32), mask, "mean"),
            "last_line": np.where(mask, x, np.inf).min(axis=1),
            "first_line": np.where(mask, x, -np.inf).max(axis=1),
        }

    h = team_shape(home)
    a = team_shape(away)
    a_own_half = 1.0 - a["own_half"]
    centroid_vx_h = np.full(h["cx"].shape, np.nan, dtype=np.float32)
    centroid_vx_a = np.full(a["cx"].shape, np.nan, dtype=np.float32)
    centroid_vx_h[12:-12] = (h["cx"][24:] - h["cx"][:-24]) * FPS / 24.0
    centroid_vx_a[12:-12] = (a["cx"][24:] - a["cx"][:-24]) * FPS / 24.0

    px, py = xy[..., 0], xy[..., 1]
    outside = ((px < -0.5) | (px > length + 0.5) | (py < -0.5) | (py > width + 0.5)) & seen_player
    corners = np.array([[0.0, 0.0], [0.0, width], [length, 0.0], [length, width]], dtype=np.float32)
    corner_distance = np.full(px.shape[0], FAR, dtype=np.float32)
    for corner in corners:
        d = np.linalg.norm(xy - corner, axis=2)
        corner_distance = np.minimum(corner_distance, np.where(seen_player, d, FAR).min(axis=1))
    centre = np.linalg.norm(xy - np.array([length / 2.0, width / 2.0], dtype=np.float32), axis=2)
    centre_distance = np.where(seen_player, centre, FAR).min(axis=1)
    in_box_h = np.sum(home & (px > length - 16.5) & (np.abs(py - width / 2.0) < 20.16), axis=1)
    in_box_a = np.sum(away & (px < 16.5) & (np.abs(py - width / 2.0) < 20.16), axis=1)

    all_mask = anyone & ~np.isnan(speed)
    gap = distance_to_observation(ball_seen)
    return {
        "ball_seen": ball_seen.astype(np.float32),
        "ball_gap": gap.astype(np.float32),
        "ball_x": bx,
        "ball_y": np.abs(by - width / 2.0),
        "ball_speed": ball_speed,
        "ball_accel": ball_accel,
        "ball_turn": ball_turn,
        "ball_edge_distance": edge.astype(np.float32),
        "ball_inside": inside,
        "all_count": np.sum(anyone, axis=1).astype(np.float32),
        "all_speed_mean": _nan_stat(speed, all_mask, "mean"),
        "all_speed_std": _nan_stat(speed, all_mask, "std"),
        "all_slow_share": _nan_stat((speed < SLOW_SPEED).astype(np.float32), all_mask, "mean"),
        "near_h": near_h,
        "near_a": near_a,
        "near_diff": near_h - near_a,
        "second_h": second_h,
        "second_a": second_a,
        "count_poss_h": np.sum(home & (distance < POSSESSION_RADIUS), axis=1).astype(np.float32),
        "count_poss_a": np.sum(away & (distance < POSSESSION_RADIUS), axis=1).astype(np.float32),
        "count_duel_h": np.sum(home & (distance < DUEL_RADIUS), axis=1).astype(np.float32),
        "count_duel_a": np.sum(away & (distance < DUEL_RADIUS), axis=1).astype(np.float32),
        "rel_speed_h": rel_h,
        "rel_speed_a": rel_a,
        "count_h": h["count"],
        "count_a": a["count"],
        "centroid_x_h": h["cx"],
        "centroid_x_a": a["cx"],
        "centroid_x_diff": h["cx"] - a["cx"],
        "centroid_y_h": np.abs(h["cy"] - width / 2.0),
        "centroid_y_a": np.abs(a["cy"] - width / 2.0),
        "spread_x_h": h["sx"],
        "spread_x_a": a["sx"],
        "spread_y_h": h["sy"],
        "spread_y_a": a["sy"],
        "spread_x_diff": h["sx"] - a["sx"],
        "spread_y_diff": h["sy"] - a["sy"],
        "speed_h": h["speed"],
        "speed_a": a["speed"],
        "slow_h": h["slow"],
        "slow_a": a["slow"],
        "own_half_h": h["own_half"],
        "own_half_a": a_own_half,
        "last_line_h": np.where(np.isfinite(h["last_line"]), h["last_line"], np.nan).astype(np.float32),
        "last_line_a": np.where(np.isfinite(a["first_line"]), length - a["first_line"], np.nan).astype(np.float32),
        "centroid_vx_h": centroid_vx_h,
        "centroid_vx_a": -centroid_vx_a,
        "ball_to_goal_h": (length - bx).astype(np.float32),
        "players_outside": np.sum(outside, axis=1).astype(np.float32),
        "corner_distance": corner_distance,
        "centre_distance": centre_distance.astype(np.float32),
        "in_box_h": in_box_h.astype(np.float32),
        "in_box_a": in_box_a.astype(np.float32),
    }


def _window_means(values: np.ndarray, low: int, high: int) -> np.ndarray:
    n = values.shape[0]
    finite = np.isfinite(values)
    cumulative = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(np.where(finite, values, 0.0).astype(np.float64), axis=0)])
    counts = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(finite, axis=0)])
    index = np.arange(n)
    lo = np.clip(index + low, 0, n)
    hi = np.clip(index + high + 1, 0, n)
    total = cumulative[hi] - cumulative[lo]
    count = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def frame_features(tracking: Tracking) -> FrameFeatures:
    blocks = []
    names: tuple[str, ...] = ()
    n = tracking.n_frames
    for start, stop in runs(tracking.present):
        base = _base_block(tracking, start, stop)
        context = np.stack([base[name] for name in CONTEXT_FEATURES], axis=1)
        extra = {}
        for width in CENTRED_WINDOWS:
            means = _window_means(context, -width, width)
            for k, name in enumerate(CONTEXT_FEATURES):
                extra[f"{name}_c{width}"] = means[:, k]
        for width in SIDED_WINDOWS:
            past = _window_means(context, -width, -1)
            future = _window_means(context, 1, width)
            for k, name in enumerate(CONTEXT_FEATURES):
                extra[f"{name}_d{width}"] = future[:, k] - past[:, k]
        merged = {**base, **extra}
        names = tuple(merged)
        blocks.append((start, stop, np.stack([merged[name] for name in names], axis=1)))
    values = np.full((n, len(names)), np.nan, dtype=np.float32)
    for start, stop, block in blocks:
        values[start:stop] = block
    return FrameFeatures(values=values, names=names)


def swap_teams(tracking: Tracking) -> Tracking:
    flipped = np.where(tracking.meta.slot_team == HOME, AWAY, HOME).astype(np.int8)
    meta = TrackingMeta(team_ids=tracking.meta.team_ids[::-1], slot_team=flipped)
    return Tracking(meta=meta, players=tracking.players, ball=tracking.ball, present=tracking.present, pitch=tracking.pitch)
