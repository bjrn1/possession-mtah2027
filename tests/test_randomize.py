import numpy as np

from possession.codes import A, D, H
from possession.external.randomize import carried_ball, random_settings, randomized, smooth_ball
from possession.io_atd import AWAY, HOME, Tracking, TrackingMeta


def toy(n: int = 300) -> tuple[Tracking, np.ndarray, np.ndarray]:
    slot_team = np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8)
    players = np.zeros((n, 4, 3), dtype=np.float32)
    players[:, :, 0] = [[20.0, 40.0, 60.0, 80.0]]
    players[:, :, 1] = 30.0
    players[100:200, 1, 1] = np.linspace(30.0, 60.0, 100)
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = 40.5
    ball[:, 1] = 30.0
    ball[50:60] = np.nan
    labels = np.full(n, H, dtype=np.int8)
    labels[100:200] = D
    labels[200:] = A
    tracking = Tracking(meta=TrackingMeta(team_ids=("h", "a"), slot_team=slot_team), players=players, ball=ball, present=np.ones(n, dtype=bool), pitch=(105.0, 68.0))
    return tracking, ball.astype(np.float64), labels


def test_smoothing_keeps_missing_frames_missing():
    ball = np.stack([np.arange(50.0), np.zeros(50)], axis=1)
    ball[10:15] = np.nan
    out = smooth_ball(ball, 7)
    assert np.isnan(out[10:15]).all()
    assert np.allclose(out[25], ball[25])


def test_carried_ball_follows_the_nearest_player():
    tracking, _, _ = toy()
    out = carried_ball(tracking.ball.astype(np.float64), tracking.players, 100, 200)
    assert np.allclose(out[150, 1], tracking.players[150, 1, 1] + 0.3)


def test_randomization_is_deterministic_and_leaves_live_play_alone_without_restore():
    tracking, clean, labels = toy()
    first = randomized(tracking, clean, labels, width=0, dead_mode="missing", dead_missing=1.0, live_restore=0.0, seed=4).ball
    second = randomized(tracking, clean, labels, width=0, dead_mode="missing", dead_missing=1.0, live_restore=0.0, seed=4).ball
    assert np.array_equal(np.isnan(first), np.isnan(second))
    assert np.isnan(first[100:170]).all()
    assert np.allclose(first[:50], tracking.ball[:50])


def test_settings_stay_inside_their_ranges():
    for seed in range(50):
        settings = random_settings(seed)
        assert 0 <= settings["width"] <= 25
        assert 0.3 <= settings["dead_missing"] <= 0.9
        assert 0.0 <= settings["live_restore"] <= 0.7
