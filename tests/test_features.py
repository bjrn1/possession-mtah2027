import numpy as np
import pytest

from possession.external.degrade import degrade, sample_params
from possession.features import distance_to_observation, frame_features, swap_teams
from tests.test_degrade import _synthetic_match


@pytest.fixture(scope="module")
def tracking():
    return degrade(_synthetic_match(n_half=1500), sample_params(2))


@pytest.fixture(scope="module")
def features(tracking):
    return frame_features(tracking)


def test_shape_and_unplayed_rows(tracking, features):
    assert features.values.shape == (tracking.n_frames, len(features.names))
    assert len(set(features.names)) == len(features.names)
    assert np.isnan(features.values[~tracking.present]).all()
    assert np.isfinite(features.values[tracking.present][:, features.names.index("near_h")]).all()


def test_swapping_teams_swaps_team_features(tracking, features):
    swapped = frame_features(swap_teams(tracking))
    index = {name: i for i, name in enumerate(features.names)}
    played = tracking.present
    for left, right in (("near_h", "near_a"), ("count_poss_h", "count_poss_a"), ("speed_h", "speed_a"), ("spread_x_h", "spread_x_a")):
        np.testing.assert_allclose(
            swapped.values[played, index[left]], features.values[played, index[right]], rtol=1e-4, atol=1e-4, equal_nan=True
        )
    np.testing.assert_allclose(
        swapped.values[played, index["near_diff"]], -features.values[played, index["near_diff"]], rtol=1e-4, atol=1e-4, equal_nan=True
    )


def test_distance_to_observation_caps_and_counts():
    seen = np.array([False, True, False, False, False, True])
    assert distance_to_observation(seen).tolist() == [1, 0, 1, 2, 1, 0]


def test_window_means_are_accurate_on_long_blocks():
    from possession.features import _window_means

    values = np.full((200000, 1), 1000.1, dtype=np.float32)
    means = _window_means(values, -5, 5)
    assert np.abs(means[-10:, 0] - 1000.1).max() < 1e-2
