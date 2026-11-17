import itertools

import numpy as np

from possession.codes import A, D, H, X
from possession.decoder import DecoderParams, _viterbi, decode, emissions, penalty_matrix


def _brute_force(scores, penalty, minimum):
    n, k = scores.shape
    best, best_path = -np.inf, None
    for path in itertools.product(range(k), repeat=n):
        segments = []
        start = 0
        for t in range(1, n + 1):
            if t == n or path[t] != path[start]:
                segments.append((path[start], t - start))
                start = t
        if any(length < minimum[label] for label, length in segments):
            continue
        total = sum(scores[t, path[t]] for t in range(n))
        total -= sum(penalty[a, b] for (a, _), (b, _) in zip(segments[:-1], segments[1:]))
        if total > best:
            best, best_path = total, path
    return np.array(best_path), best


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = 8
        scores = rng.normal(0.0, 1.5, (n, 3))
        penalty = rng.uniform(0.0, 3.0, (3, 3))
        np.fill_diagonal(penalty, 0.0)
        minimum = rng.integers(1, 4, 3).astype(np.int64)
        expected, expected_score = _brute_force(scores, penalty, minimum)
        got = _viterbi(scores, penalty, minimum, np.zeros(n))
        segments = [(got[0], 1)]
        for label in got[1:]:
            if label == segments[-1][0]:
                segments[-1] = (label, segments[-1][1] + 1)
            else:
                segments.append((label, 1))
        assert all(length >= minimum[label] for label, length in segments)
        got_score = sum(scores[t, got[t]] for t in range(n)) - sum(
            penalty[a, b] for (a, _), (b, _) in zip(segments[:-1], segments[1:])
        )
        assert np.isclose(got_score, expected_score), (trial, got, expected)


def test_short_flicker_is_removed_and_halves_are_separate():
    probabilities = np.zeros((300, 2), dtype=np.float32)
    probabilities[:, 0] = 0.02
    probabilities[:, 1] = 0.1
    probabilities[100:104, 1] = 0.9
    probabilities[200:, 1] = 0.95
    played = np.ones(300, dtype=bool)
    played[150] = False
    labels = decode(probabilities, played, DecoderParams(min_possession_frames=12))
    assert (labels[:150] == H).all()
    assert labels[150] == X
    assert (labels[205:] == A).all()


def test_emissions_order():
    scores = emissions(np.array([[0.9, 0.5], [0.1, 0.9]]), DecoderParams())
    assert np.argmax(scores[0]) == D
    assert np.argmax(scores[1]) == A
    assert penalty_matrix(DecoderParams())[H, H] == 0.0


def test_boundary_bonus_enables_short_possession():
    probabilities = np.zeros((200, 2), dtype=np.float32)
    probabilities[:, 0] = 0.02
    probabilities[:, 1] = 0.3
    probabilities[100:120, 1] = 0.7
    played = np.ones(200, dtype=bool)
    params = DecoderParams(switch_penalty=20.0, min_possession_frames=6)
    assert (decode(probabilities, played, params) == H).all()
    boundary = np.full(200, 0.01, dtype=np.float32)
    boundary[[100, 120]] = 0.99
    boosted = decode(probabilities, played, DecoderParams(switch_penalty=20.0, min_possession_frames=6, boundary_weight=4.0), boundary)
    assert (boosted[100:120] == A).all()
    assert (boosted[:100] == H).all()


def test_possession_weights_count_each_possession_once():
    from possession.external.train import possession_weights

    labels = np.array([H] * 100 + [A] * 30 + [D] * 10 + [H] * 5, dtype=np.int8)
    weights = possession_weights(labels)
    assert np.isclose(weights[:100].sum(), 1.0)
    assert np.isclose(weights[100:130].sum(), 1.0)
    assert (weights[130:140] == 0).all()
    assert np.isclose(weights[140:].sum(), 5 / 25)


def test_short_block_respects_clipped_minimum_instead_of_argmax():
    probabilities = np.zeros((30, 2), dtype=np.float32)
    probabilities[:, 0] = 0.02
    probabilities[:, 1] = np.where(np.arange(30) % 2 == 0, 0.6, 0.4)
    labels = decode(probabilities, np.ones(30, dtype=bool), DecoderParams(min_possession_frames=12, min_dead_frames=50))
    changes = np.count_nonzero(labels[1:] != labels[:-1])
    assert changes <= 2
