import numpy as np

from possession.codes import runs


def linear_fill(ball: np.ndarray, present: np.ndarray) -> np.ndarray:
    out = np.full(ball.shape, np.nan)
    for start, stop in runs(present):
        block = ball[start:stop]
        seen = ~np.isnan(block[:, 0])
        if seen.sum() < 2:
            continue
        index = np.arange(stop - start)
        for axis in range(2):
            out[start:stop, axis] = np.interp(index, index[seen], block[seen, axis])
    return out


def remove_spikes(ball: np.ndarray, present: np.ndarray, max_speed: float = 40.0, fps: int = 25) -> np.ndarray:
    out = ball.astype(np.float64).copy()
    out[~present] = np.nan
    for start, stop in runs(present):
        block = out[start:stop]
        seen = np.flatnonzero(~np.isnan(block[:, 0]))
        if seen.size < 3:
            continue
        points = block[seen]
        dt = np.diff(seen) / fps
        step = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
        fast = step > max_speed
        spike = np.zeros(seen.size, dtype=bool)
        spike[1:-1] = fast[:-1] & fast[1:]
        spike[0] = fast[0] and not fast[1]
        spike[-1] = fast[-1] and not fast[-2]
        block[seen[spike]] = np.nan
    return out
