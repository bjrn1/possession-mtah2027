import numpy as np

from possession.features import swap_teams
from possession.features_control import NAMES, control_features
from possession.io_atd import AWAY, HOME, Tracking, TrackingMeta

TEAM_SIGNED = ("ball_last_near_diff", "ball_next_near_diff", "control_now", "control_last_team", "control_next_team", "touch_last_team", "touch_next_team", "control_soft_c12", "control_soft_c50", "control_soft_c125")


def toy_tracking(n: int = 400) -> Tracking:
    slot_team = np.array([HOME] * 3 + [AWAY] * 3, dtype=np.int8)
    players = np.zeros((n, 6, 3), dtype=np.float32)
    t = np.arange(n)
    players[:, 0, 0] = 30.0 + 0.2 * t
    players[:, 0, 1] = 34.0
    players[:, 1:3, 0] = [[20.0, 25.0]]
    players[:, 1:3, 1] = [[20.0, 50.0]]
    players[:, 3:, 0] = [[70.0, 80.0, 90.0]]
    players[:, 3:, 1] = [[30.0, 40.0, 50.0]]
    players[200:, 3, 0] = players[200:, 0, 0] + 0.5
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = players[:, 0, 0] + 0.3
    ball[:, 1] = 34.0
    ball[200:, 0] = players[200:, 3, 0] + 0.3
    ball[120:160] = np.nan
    return Tracking(meta=TrackingMeta(team_ids=("h", "a"), slot_team=slot_team), players=players, ball=ball, present=np.ones(n, dtype=bool), pitch=(105.0, 68.0))


def test_shape_and_names():
    values = control_features(toy_tracking())
    assert values.shape == (400, len(NAMES))


def test_ball_gap_timers():
    values = control_features(toy_tracking())
    since = values[:, NAMES.index("ball_since_seen")]
    until = values[:, NAMES.index("ball_until_seen")]
    assert since[100] == 0.0
    assert np.isclose(since[130], 11 / 25)
    assert np.isclose(until[130], 30 / 25)


def test_home_control_is_detected_before_the_change():
    values = control_features(toy_tracking())
    control = values[:, NAMES.index("control_now")]
    assert (control[20:110] == 1.0).mean() > 0.9


def test_team_signed_features_flip_when_teams_are_swapped():
    tracking = toy_tracking()
    original = control_features(tracking)
    swapped = control_features(swap_teams(tracking))
    for name in TEAM_SIGNED:
        column = NAMES.index(name)
        assert np.allclose(np.nan_to_num(original[:, column]), -np.nan_to_num(swapped[:, column])), name
