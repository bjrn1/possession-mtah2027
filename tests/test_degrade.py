import numpy as np
import pytest

from possession.external.degrade import DegradeRanges, degrade, sample_params
from possession.external.dfl import ExternalMatch


def _synthetic_match(n_half: int = 3000, seed: int = 7) -> ExternalMatch:
    rng = np.random.default_rng(seed)
    n_players = 24
    n_frames = 2 * n_half + 1
    half = np.concatenate([np.ones(n_half), [0], np.full(n_half, 2)]).astype(np.int8)
    base = rng.uniform([5.0, 5.0], [100.0, 63.0], (n_players, 2))
    walk = np.cumsum(rng.normal(0.0, 0.05, (n_frames, n_players, 2)), axis=0)
    xy = np.clip(base[None] + walk, 0.0, [105.0, 68.0]).astype(np.float32)
    players = np.concatenate([xy, np.ones((n_frames, n_players, 1), dtype=np.float32)], axis=2)
    players[:, 22:] = np.nan
    players[half == 0] = np.nan
    ball = players[:, 0, :2] + 0.3
    ball_z = np.zeros(n_frames, dtype=np.float32)
    return ExternalMatch(
        match_id="SYN",
        pitch=(105.0, 68.0),
        players=players,
        slot_team=np.array([0] * 12 + [1] * 12, dtype=np.int8),
        ball=ball.astype(np.float32),
        ball_z=ball_z,
        half=half,
        frame_n=np.arange(n_frames, dtype=np.int64),
        ball_status=np.ones(n_frames, dtype=np.int8),
        ball_possession=np.ones(n_frames, dtype=np.int8),
    )


def test_params_are_deterministic_and_inside_ranges():
    first = sample_params(11)
    second = sample_params(11)
    assert first == second
    ranges = DegradeRanges()
    assert ranges.ball_missing[0] <= first.ball_missing <= ranges.ball_missing[1]


def test_degradation_hits_target_ball_missing_share():
    match = _synthetic_match()
    params = sample_params(3)
    tracking = degrade(match, params)
    played = match.half > 0
    share = np.isnan(tracking.ball[played, 0]).mean()
    assert share == pytest.approx(params.ball_missing, abs=0.03)
    assert np.isnan(tracking.ball[match.half == 0]).all()


def test_degradation_is_reproducible():
    match = _synthetic_match()
    params = sample_params(5)
    first = degrade(match, params)
    second = degrade(match, params)
    np.testing.assert_array_equal(first.players, second.players)
    np.testing.assert_array_equal(first.ball, second.ball)


def test_slots_carry_team_and_drop_identity():
    match = _synthetic_match()
    tracking = degrade(match, sample_params(9))
    assert tracking.meta.slot_team.size > 22
    assert set(np.unique(tracking.meta.slot_team).tolist()) <= {0, 1}
    played = match.half > 0
    tracked = np.sum(~np.isnan(tracking.players[played, :, 0]), axis=1)
    assert 15 < tracked.mean() < 22.5


def test_speed_is_computed_from_slot_positions_only():
    match = _synthetic_match()
    tracking = degrade(match, sample_params(4))
    speed = tracking.players[:, :, 2]
    seen = ~np.isnan(tracking.players[:, :, 0])
    finite = np.isfinite(speed)
    frames, slots = np.nonzero(finite)
    assert seen[frames - 2, slots].all() and seen[frames + 2, slots].all()


def test_duplicate_slots_are_not_grouped_at_the_end():
    from dataclasses import replace

    match = _synthetic_match(n_half=6000)
    params = replace(sample_params(8), duplicate_rate_per_minute=6.0)
    tracking = degrade(match, params)
    occupancy = (~np.isnan(tracking.players[:, :, 0])).sum(axis=0)
    tail = occupancy[-5:]
    assert tail.min() > 0 and not (tail < 200).all()
