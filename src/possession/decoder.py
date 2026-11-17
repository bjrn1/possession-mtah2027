from dataclasses import dataclass

import numba
import numpy as np

from possession.codes import A, D, H, X, runs

LABELS = (H, A, D)
EPSILON = 1e-4


@dataclass(frozen=True)
class DecoderParams:
    switch_penalty: float = 8.0
    stop_penalty: float = 6.0
    restart_penalty: float = 6.0
    min_possession_frames: int = 12
    min_dead_frames: int = 50
    dead_bias: float = 0.0
    emission_weight: float = 1.0
    boundary_weight: float = 0.0


def emissions(probabilities: np.ndarray, params: DecoderParams) -> np.ndarray:
    dead = np.clip(probabilities[:, 0], EPSILON, 1.0 - EPSILON).astype(np.float64)
    away = np.clip(probabilities[:, 1], EPSILON, 1.0 - EPSILON).astype(np.float64)
    out = np.empty((probabilities.shape[0], 3))
    out[:, H] = np.log1p(-dead) + np.log1p(-away)
    out[:, A] = np.log1p(-dead) + np.log(away)
    out[:, D] = np.log(dead) + params.dead_bias
    return out * params.emission_weight


@numba.njit(cache=True)
def _viterbi(scores: np.ndarray, penalty: np.ndarray, minimum: np.ndarray, bonus: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    k = scores.shape[1]
    width = minimum.max()
    neg = -1e18
    chain = np.full((k, width), neg)
    entry_from = np.full((n, k), -1, dtype=np.int8)
    last_from_self = np.zeros((n, k), dtype=np.bool_)
    for label in range(k):
        chain[label, 0] = scores[0, label]
    for t in range(1, n):
        previous = chain.copy()
        for label in range(k):
            best = neg
            best_from = -1
            for other in range(k):
                if other == label:
                    continue
                candidate = previous[other, minimum[other] - 1] - penalty[other, label] + bonus[t]
                if candidate > best:
                    best = candidate
                    best_from = other
            m = minimum[label]
            if m == 1:
                stay = previous[label, 0]
                if stay >= best:
                    chain[label, 0] = stay + scores[t, label]
                    last_from_self[t, label] = True
                else:
                    chain[label, 0] = best + scores[t, label]
                    entry_from[t, label] = best_from
                continue
            chain[label, 0] = best + scores[t, label]
            entry_from[t, label] = best_from
            for position in range(1, m - 1):
                chain[label, position] = previous[label, position - 1] + scores[t, label]
            stay = previous[label, m - 1]
            advance = previous[label, m - 2]
            if stay >= advance:
                chain[label, m - 1] = stay + scores[t, label]
                last_from_self[t, label] = True
            else:
                chain[label, m - 1] = advance + scores[t, label]
    out = np.empty(n, dtype=np.int8)
    label = 0
    best = neg
    for candidate in range(k):
        if chain[candidate, minimum[candidate] - 1] > best:
            best = chain[candidate, minimum[candidate] - 1]
            label = candidate
    position = minimum[label] - 1
    t = n - 1
    while t >= 0:
        out[t] = label
        if t == 0:
            break
        m = minimum[label]
        if position == m - 1 and m > 1:
            if last_from_self[t, label]:
                position = m - 1
            else:
                position = m - 2
        elif position == m - 1 and m == 1:
            if not last_from_self[t, label]:
                label = entry_from[t, label]
                position = minimum[label] - 1
        elif position > 0:
            position -= 1
        else:
            label = entry_from[t, label]
            position = minimum[label] - 1
        t -= 1
    return out


def penalty_matrix(params: DecoderParams) -> np.ndarray:
    penalty = np.zeros((3, 3))
    penalty[H, A] = penalty[A, H] = params.switch_penalty
    penalty[H, D] = penalty[A, D] = params.stop_penalty
    penalty[D, H] = penalty[D, A] = params.restart_penalty
    return penalty


def boundary_bonus(boundary: np.ndarray | None, size: int, params: DecoderParams) -> np.ndarray:
    if boundary is None or params.boundary_weight == 0.0:
        return np.zeros(size)
    p = np.clip(np.nan_to_num(boundary, nan=EPSILON), EPSILON, 1.0 - EPSILON).astype(np.float64)
    return params.boundary_weight * (np.log(p) - np.log1p(-p))


def decode(
    probabilities: np.ndarray,
    played: np.ndarray,
    params: DecoderParams = DecoderParams(),
    boundary: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.full(probabilities.shape[0], X, dtype=np.int8)
    penalty = penalty_matrix(params)
    minimum = np.array(
        [params.min_possession_frames, params.min_possession_frames, params.min_dead_frames], dtype=np.int64
    )
    for start, stop in runs(played):
        scores = emissions(probabilities[start:stop], params)
        block_minimum = np.minimum(minimum, stop - start)
        bonus = boundary_bonus(None if boundary is None else boundary[start:stop], stop - start, params)
        labels[start:stop] = _viterbi(scores, penalty, block_minimum, bonus)
    return labels
