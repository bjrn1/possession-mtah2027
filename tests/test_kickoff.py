import numpy as np

from possession.codes import A, D, H
from possession.io_atd import AWAY, HOME, Tracking, TrackingMeta
from possession.kickoff import dead_until_kickoff, lineup_runs

LENGTH, WIDTH = 105.0, 68.0


def tracking_with_lineup(n: int, lineup_end: int, walking_before: bool) -> Tracking:
    rng = np.random.default_rng(3)
    slot_team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    players = np.zeros((n, 22, 3), dtype=np.float32)
    base = np.concatenate([np.stack([rng.uniform(10, 45, 11), rng.uniform(5, 63, 11)], axis=1), np.stack([rng.uniform(60, 95, 11), rng.uniform(5, 63, 11)], axis=1)])
    players[:, :, :2] = base[None]
    before = lineup_end - 300
    drift = np.arange(before)[:, None] / before
    mixed = np.where(slot_team == HOME, 45.0, -45.0)[None, :]
    players[:before, :, 0] += mixed * (1.0 - drift)
    if not walking_before:
        players[:before, :, 0] += 8.0 * np.sin(np.arange(before) / 3.0)[:, None]
    running = np.arange(n - lineup_end)[:, None]
    players[lineup_end:, :, 0] += 0.4 * running * np.where(slot_team == HOME, 1.0, -1.0)[None, :] % 30.0
    players[lineup_end:, :, 1] += 3.0 * np.sin(running / 4.0)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    return Tracking(meta=TrackingMeta(team_ids=("h", "a"), slot_team=slot_team), players=players, ball=ball, present=np.ones(n, dtype=bool), pitch=(LENGTH, WIDTH))


def test_lineup_is_found_when_teams_stand_in_their_halves():
    tracking = tracking_with_lineup(3000, 2000, walking_before=True)
    runs, _ = lineup_runs(tracking, 0, 3000)
    assert runs
    assert runs[-1][1] >= 1990


def test_frames_before_the_kickoff_become_dead():
    tracking = tracking_with_lineup(3000, 2000, walking_before=True)
    labels = np.full(3000, H, dtype=np.int8)
    labels[500:900] = A
    dead = np.full(3000, 0.9)
    out = dead_until_kickoff(labels, tracking, dead, [(100, 3000)])
    runs, _ = lineup_runs(tracking, 0, 3000)
    kick = runs[-1][1]
    assert (out[100:kick] == D).all()
    assert (out[kick:] == labels[kick:]).all()


def test_half_that_opens_inside_the_lineup_is_untouched():
    tracking = tracking_with_lineup(3000, 2000, walking_before=True)
    labels = np.full(3000, H, dtype=np.int8)
    out = dead_until_kickoff(labels, tracking, np.full(3000, 0.9), [(1500, 3000)])
    assert (out == labels).all()


def test_live_play_before_a_lineup_is_never_made_dead():
    tracking = tracking_with_lineup(3000, 2000, walking_before=False)
    labels = np.full(3000, H, dtype=np.int8)
    out = dead_until_kickoff(labels, tracking, np.full(3000, 0.1), [(0, 3000)])
    assert (out == labels).all()
