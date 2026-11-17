import numpy as np

from possession.codes import D, runs
from possession.features import FPS, home_attacks_right
from possession.io_atd import HOME, Tracking

LINEUP_FRAMES = 50
LINEUP_SHARE = 0.9
LINEUP_LINE_M = 2.0
LINEUP_MIN_PLAYERS = 7
WALK_MS = 1.5
SEARCH_FRAMES = 180 * FPS
DEAD_PROBABILITY = 0.5


def _player_speed(xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape[:2], np.nan, dtype=np.float64)
    out[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) * FPS / 4.0
    return out


def lineup_runs(tracking: Tracking, start: int, stop: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    length = tracking.pitch[0]
    xy = tracking.players[start:stop, :, :2].astype(np.float64)
    if not home_attacks_right(tracking, start, stop):
        xy[..., 0] = length - xy[..., 0]
    seen = ~np.isnan(xy[..., 0])
    x = np.where(seen, xy[..., 0], np.nan)
    home = tracking.meta.slot_team == HOME
    half = length / 2.0
    with np.errstate(invalid="ignore"):
        home_own = (x[:, home] <= half + LINEUP_LINE_M) & seen[:, home]
        away_own = (x[:, ~home] >= half - LINEUP_LINE_M) & seen[:, ~home]
    home_count = seen[:, home].sum(axis=1)
    away_count = seen[:, ~home].sum(axis=1)
    speed = _player_speed(xy)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.nansum(speed, axis=1) / np.maximum(np.sum(~np.isnan(speed), axis=1), 1)
        lined = (
            (home_count >= LINEUP_MIN_PLAYERS)
            & (away_count >= LINEUP_MIN_PLAYERS)
            & (home_own.sum(axis=1) >= LINEUP_SHARE * home_count)
            & (away_own.sum(axis=1) >= LINEUP_SHARE * away_count)
            & (mean_speed < WALK_MS)
        )
    found = [(start + a, start + b) for a, b in runs(lined) if b - a >= LINEUP_FRAMES]
    return found, mean_speed


def dead_until_kickoff(labels: np.ndarray, tracking: Tracking, dead_probability: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    out = labels.copy()
    for first, stop in blocks:
        context_start = max(0, first - SEARCH_FRAMES)
        context_stop = min(stop, first + SEARCH_FRAMES)
        lineups, mean_speed = lineup_runs(tracking, context_start, context_stop)
        if any(a <= first < b for a, b in lineups):
            continue
        current = first
        for _, end in lineups:
            if not first < end <= min(stop, first + SEARCH_FRAMES):
                continue
            stretch = slice(current, end)
            local = slice(current - context_start, end - context_start)
            walking = np.nanmean(mean_speed[local]) < WALK_MS if np.isfinite(mean_speed[local]).any() else False
            dead = np.nanmean(dead_probability[stretch]) >= DEAD_PROBABILITY if np.isfinite(dead_probability[stretch]).any() else False
            if not (walking or dead):
                break
            current = end
        if current > first:
            out[first:current] = D
    return out
