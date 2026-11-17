import numpy as np

from possession.temporal import BLOCKS, STRIDE, _conv, class_probabilities, dead_and_away


def random_network(n_features: int, channels: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    network = {
        "mean": rng.normal(size=n_features),
        "std": rng.uniform(0.5, 2.0, size=n_features),
        "0_weight": rng.normal(scale=0.3, size=(channels, 2 * n_features, 1)),
        "0_bias": rng.normal(scale=0.1, size=channels),
        f"{BLOCKS + 1}_weight": rng.normal(scale=0.3, size=(3, channels, 1)),
        f"{BLOCKS + 1}_bias": rng.normal(scale=0.1, size=3),
    }
    for block in range(BLOCKS):
        for layer in ("0", "3"):
            network[f"{block + 1}_conv_{layer}_weight"] = rng.normal(scale=0.3, size=(channels, channels, 3))
            network[f"{block + 1}_conv_{layer}_bias"] = rng.normal(scale=0.1, size=channels)
    return network


def test_dilated_convolution_matches_a_direct_sum():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 20))
    weight = rng.normal(size=(2, 3, 3))
    bias = rng.normal(size=2)
    dilation = 4
    out = _conv(x, weight, bias, dilation)
    expected = np.zeros((2, 20))
    for t in range(20):
        for k, offset in enumerate((-dilation, 0, dilation)):
            if 0 <= t + offset < 20:
                expected[:, t] += weight[:, :, k] @ x[:, t + offset]
        expected[:, t] += bias
    assert np.allclose(out, expected)


def test_class_probabilities_sum_to_one_and_cover_every_frame():
    network = random_network(5)
    features = np.random.default_rng(2).normal(size=(53, 5)).astype(np.float32)
    features[7, 2] = np.nan
    probabilities = class_probabilities(network, features)
    assert probabilities.shape == (53, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()


def test_each_phase_is_predicted_independently():
    network = random_network(5)
    features = np.random.default_rng(4).normal(size=(50, 5)).astype(np.float32)
    full = class_probabilities(network, features)
    changed = features.copy()
    changed[1::STRIDE] += 10.0
    other = class_probabilities(network, changed)
    assert np.allclose(full[0::STRIDE], other[0::STRIDE])
    assert not np.allclose(full[1::STRIDE], other[1::STRIDE])


def test_dead_and_away_maps_three_classes_to_two_channels():
    network = random_network(5)
    features = np.random.default_rng(5).normal(size=(40, 5)).astype(np.float32)
    three = class_probabilities(network, features)
    two = dead_and_away(network, features)
    assert np.allclose(two[:, 0], three[:, 2])
    assert np.allclose(two[:, 1], three[:, 1] / (three[:, 0] + three[:, 1]))
