import numpy as np

from possession.baseline import baseline_labels, majority_smooth, nearest_team
from possession.codes import A, D, H, X
from possession.io_atd import AWAY, HOME, Tracking, TrackingMeta


def _tracking() -> Tracking:
    players = np.full((6, 2, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = [10.0, 10.0]
    players[:, 1, :2] = [20.0, 10.0]
    ball = np.array([[11, 10], [12, 10], [19, 10], [np.nan, np.nan], [19, 10], [11, 10]], dtype=np.float32)
    meta = TrackingMeta(team_ids=("h", "a"), slot_team=np.array([HOME, AWAY], dtype=np.int8))
    return Tracking(meta=meta, players=players, ball=ball, present=np.ones(6, dtype=bool), pitch=(105.0, 68.0))


def test_nearest_team_marks_missing_ball():
    assert nearest_team(_tracking()).tolist() == [HOME, HOME, AWAY, -1, AWAY, HOME]


def test_raw_labels_use_dead_ball_for_missing():
    assert majority_smooth(nearest_team(_tracking()), 0).tolist() == [H, H, A, D, A, H]


def test_smoothing_votes_and_ties_go_home():
    assert majority_smooth(nearest_team(_tracking()), 1).tolist() == [H, H, H, A, H, H]


def test_unplayed_frames_are_abstained():
    played = np.array([True, True, True, True, True, False])
    assert baseline_labels(_tracking(), 0, played)[-1] == X


def test_smoothing_does_not_cross_unplayed_frames():
    players = np.full((7, 2, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = [10.0, 10.0]
    players[:, 1, :2] = [20.0, 10.0]
    ball = np.array([[19, 10]] * 3 + [[np.nan, np.nan]] + [[11, 10]] * 3, dtype=np.float32)
    meta = TrackingMeta(team_ids=("h", "a"), slot_team=np.array([HOME, AWAY], dtype=np.int8))
    tracking = Tracking(meta=meta, players=players, ball=ball, present=np.ones(7, dtype=bool), pitch=(105.0, 68.0))
    played = np.array([True, True, True, False, True, True, True])
    labels = baseline_labels(tracking, 5, played)
    assert labels.tolist() == [A, A, A, X, H, H, H]
