import numpy as np

from possession.codes import D, runs
from possession.io_atd import Tracking

SMOOTHING_WIDTHS = (0, 0, 3, 7, 13, 25)
DEAD_MODES = ("keep", "missing", "carried", "mixed")
CARRY_OFFSET_M = (0.4, 0.3)
KEPT_BEFORE_RESTART_FRAMES = 25


def smooth_ball(ball: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return ball
    seen = ~np.isnan(ball[:, 0])
    values = np.where(seen[:, None], ball, 0.0)
    kernel = np.ones(width)
    total = np.stack([np.convolve(values[:, k], kernel, mode="same") for k in range(2)], axis=1)
    count = np.convolve(seen.astype(float), kernel, mode="same")
    out = np.full_like(ball, np.nan)
    out[seen] = total[seen] / count[seen, None]
    return out


def carried_ball(ball: np.ndarray, players: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = ball.copy()
    segment = ball[start:stop]
    finite = ~np.isnan(segment[:, 0])
    if start > 0 and not np.isnan(ball[start - 1, 0]):
        anchor = ball[start - 1]
    elif finite.any():
        anchor = segment[finite].mean(axis=0)
    else:
        return out
    distance = np.linalg.norm(players[start, :, :2] - anchor, axis=1)
    if not np.isfinite(distance).any():
        return out
    slot = int(np.argmin(np.where(np.isfinite(distance), distance, np.inf)))
    out[start:stop] = players[start:stop, slot, :2].astype(np.float64) + np.array(CARRY_OFFSET_M)
    return out


def randomized(tracking: Tracking, clean_ball: np.ndarray, labels: np.ndarray, width: int, dead_mode: str, dead_missing: float, live_restore: float, seed: int) -> Tracking:
    rng = np.random.default_rng(seed)
    ball = tracking.ball.astype(np.float64).copy()
    played = tracking.present
    if live_restore > 0:
        alive = (labels == 0) | (labels == 1)
        missing_live = np.isnan(ball[:, 0]) & alive & played & ~np.isnan(clean_ball[:, 0])
        for start, stop in runs(missing_live):
            if rng.random() < live_restore:
                ball[start:stop] = clean_ball[start:stop] + rng.normal(0.0, 0.05, size=(stop - start, 2))
    for start, stop in runs((labels == D) & played):
        mode = dead_mode if dead_mode != "mixed" else ("missing" if rng.random() < 0.5 else "carried")
        if mode == "missing" and rng.random() < dead_missing:
            kept = min(stop - start, int(rng.integers(0, KEPT_BEFORE_RESTART_FRAMES)))
            ball[start:stop - kept] = np.nan
        elif mode == "carried":
            ball = carried_ball(ball, tracking.players, start, stop)
    ball = smooth_ball(ball, width)
    ball[~played] = np.nan
    return Tracking(meta=tracking.meta, players=tracking.players, ball=ball.astype(np.float32), present=tracking.present, pitch=tracking.pitch)


def random_settings(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "width": int(rng.choice(SMOOTHING_WIDTHS)),
        "dead_mode": str(rng.choice(DEAD_MODES)),
        "dead_missing": float(rng.uniform(0.3, 0.9)),
        "live_restore": float(rng.choice([0.0, 0.0, rng.uniform(0.2, 0.7)])),
    }
