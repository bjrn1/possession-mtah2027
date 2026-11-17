import numpy as np
import pytest

from possession.ball import linear_fill, remove_spikes


def _straight_ball(n: int = 200, speed: float = 10.0) -> np.ndarray:
    t = np.arange(n) / 25.0
    return np.stack([10.0 + speed * t, np.full(n, 30.0)], axis=1)


def test_linear_fill():
    ball = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 4.0]])
    assert linear_fill(ball, np.ones(3, dtype=bool))[1] == pytest.approx([1.0, 2.0])


def test_linear_fill_keeps_unplayed_frames_empty():
    ball = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 4.0], [3.0, 3.0]])
    played = np.array([True, True, True, False])
    out = linear_fill(ball, played)
    assert np.isnan(out[3]).all()


def test_remove_spikes_drops_isolated_jump_only():
    truth = _straight_ball(50)
    ball = truth.copy()
    ball[20] = [90.0, 5.0]
    cleaned = remove_spikes(ball, np.ones(50, dtype=bool))
    assert np.isnan(cleaned[20]).all()
    assert np.isfinite(cleaned[[19, 21]]).all()


def test_remove_spikes_handles_block_edges():
    truth = _straight_ball(50)
    ball = truth.copy()
    ball[0] = [95.0, 60.0]
    ball[-1] = [0.0, 0.0]
    cleaned = remove_spikes(ball, np.ones(50, dtype=bool))
    assert np.isnan(cleaned[0]).all()
    assert np.isnan(cleaned[-1]).all()
    assert np.isfinite(cleaned[1:-1]).all()
