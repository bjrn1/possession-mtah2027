from pathlib import Path

import numpy as np

STRIDE = 5
BLOCKS = 6


def load_network(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float64) for key in data.files}


def _conv(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int) -> np.ndarray:
    length = x.shape[1]
    padded = np.zeros((x.shape[0], length + 2 * dilation))
    padded[:, dilation:dilation + length] = x
    out = weight[:, :, 0] @ padded[:, :length]
    out += weight[:, :, 1] @ padded[:, dilation:dilation + length]
    out += weight[:, :, 2] @ padded[:, 2 * dilation:2 * dilation + length]
    return out + bias[:, None]


def _forward(network: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = network["0_weight"][:, :, 0] @ x + network["0_bias"][:, None]
    for block in range(BLOCKS):
        prefix = f"{block + 1}_conv_"
        dilation = 2 ** block
        inner = np.maximum(_conv(h, network[prefix + "0_weight"], network[prefix + "0_bias"], dilation), 0.0)
        inner = np.maximum(_conv(inner, network[prefix + "3_weight"], network[prefix + "3_bias"], dilation), 0.0)
        h = h + inner
    last = f"{BLOCKS + 1}_"
    logits = network[last + "weight"][:, :, 0] @ h + network[last + "bias"][:, None]
    logits -= logits.max(axis=0, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=0, keepdims=True)


def class_probabilities(network: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    mean = network["mean"]
    std = network["std"]
    out = np.empty((features.shape[0], 3))
    for phase in range(STRIDE):
        x = features[phase::STRIDE].astype(np.float64)
        missing = np.isnan(x)
        z = np.where(missing, 0.0, (x - mean) / std)
        inputs = np.concatenate([z, missing.astype(np.float64)], axis=1).T
        out[phase::STRIDE] = _forward(network, inputs).T
    return out


def dead_and_away(network: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    probabilities = class_probabilities(network, features)
    alive = np.maximum(probabilities[:, 0] + probabilities[:, 1], 1e-12)
    return np.stack([probabilities[:, 2], probabilities[:, 1] / alive], axis=1)
