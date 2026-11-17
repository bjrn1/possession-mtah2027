import numpy as np

from possession.codes import A, D, H, X, runs
from possession.io_atd import AWAY, HOME, Tracking


def nearest_team(tracking: Tracking) -> np.ndarray:
    xy = tracking.players[:, :, :2]
    distance = np.linalg.norm(xy - tracking.ball[:, None, :], axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(distance.shape[0]), nearest])
    team = tracking.meta.slot_team[nearest].astype(np.int8)
    out = np.full(tracking.n_frames, -1, dtype=np.int8)
    out[found] = team[found]
    return out


def majority_smooth(team: np.ndarray, half_width: int) -> np.ndarray:
    if half_width <= 0:
        return np.where(team == HOME, H, np.where(team == AWAY, A, D)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(team == HOME)))
    away = np.concatenate(([0], np.cumsum(team == AWAY)))
    index = np.arange(team.size)
    low = np.maximum(0, index - half_width)
    high = np.minimum(team.size, index + half_width + 1)
    home_votes = home[high] - home[low]
    away_votes = away[high] - away[low]
    out = np.full(team.size, D, dtype=np.int8)
    seen = home_votes + away_votes > 0
    out[seen] = np.where(home_votes[seen] >= away_votes[seen], H, A)
    return out


def baseline_labels(tracking: Tracking, half_width: int, played: np.ndarray) -> np.ndarray:
    team = nearest_team(tracking)
    labels = np.full(tracking.n_frames, X, dtype=np.int8)
    for start, stop in runs(played):
        labels[start:stop] = majority_smooth(team[start:stop], half_width)
    return labels
