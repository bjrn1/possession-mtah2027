import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from possession.external.make_features import feature_path
from possession.external.train import MATCHES, VARIANTS
from possession.temporal import STRIDE

CHANNELS = 64
BLOCKS = 6
KERNEL = 3
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH = 8
CROP = 1024
STEPS = 1500
THREADS = 1


def network(torch, n_inputs: int):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(CHANNELS, CHANNELS, KERNEL, dilation=dilation, padding=dilation), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Conv1d(CHANNELS, CHANNELS, KERNEL, dilation=dilation, padding=dilation), nn.ReLU(), nn.Dropout(DROPOUT))

        def forward(self, x):
            return x + self.conv(x)

    return nn.Sequential(nn.Conv1d(n_inputs, CHANNELS, 1), *[Block(2 ** i) for i in range(BLOCKS)], nn.Conv1d(CHANNELS, 3, 1))


def sequence(cache_dir: Path, name: str, swapped: bool, label: str, phase: int = 0, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    base = np.load(feature_path(cache_dir, name, swapped, "x"), mmap_mode="r")[phase::stride]
    control = np.load(feature_path(cache_dir, name, swapped, "c"), mmap_mode="r")[phase::stride]
    x = np.concatenate([np.asarray(base, dtype=np.float32), np.asarray(control, dtype=np.float32)], axis=1)
    y = np.load(feature_path(cache_dir, name, swapped, label))[phase::stride].astype(np.int64)
    return x, np.where(y > 2, -1, y)


def standardised(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    missing = np.isnan(x)
    z = np.where(missing, 0.0, (x - mean) / std).astype(np.float32)
    return np.concatenate([z, missing.astype(np.float32)], axis=1)


def fit(torch, cache_dir: Path, matches: list[str], label: str, with_randomized: bool):
    sequences = []
    for index, match_id in enumerate(matches):
        for variant in VARIANTS:
            for swapped in (False, True):
                phase = (index + variant + int(swapped)) % STRIDE
                x, y = sequence(cache_dir, f"{match_id}_v{variant:02d}", swapped, label, phase, STRIDE)
                sequences.append((x.astype(np.float16), y))
                if with_randomized:
                    x, y = sequence(cache_dir, f"dr_{match_id}_v{variant:02d}", swapped, label, (phase + 2) % STRIDE, STRIDE)
                    sequences.append((x.astype(np.float16), y))
    width = sequences[0][0].shape[1]
    total = np.zeros(width)
    square = np.zeros(width)
    count = np.zeros(width)
    for x16, _ in sequences:
        x = x16.astype(np.float32)
        finite = np.isfinite(x)
        total += np.where(finite, x, 0.0).sum(axis=0)
        square += np.where(finite, x * x, 0.0).sum(axis=0)
        count += finite.sum(axis=0)
    mean = (total / np.maximum(count, 1)).astype(np.float32)
    std = (np.sqrt(np.maximum(square / np.maximum(count, 1) - mean * mean, 0.0)) + 1e-6).astype(np.float32)
    data = sequences
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = network(torch, 2 * width)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    for _ in range(STEPS):
        xs, ys = [], []
        for _ in range(BATCH):
            x16, y = data[rng.integers(len(data))]
            start = int(rng.integers(0, len(y) - CROP))
            xs.append(torch.from_numpy(standardised(x16[start:start + CROP].astype(np.float32), mean, std)).T)
            ys.append(torch.from_numpy(y[start:start + CROP]))
        optimiser.zero_grad()
        loss = loss_fn(model(torch.stack(xs)), torch.stack(ys))
        loss.backward()
        optimiser.step()
    model.eval()
    return model, mean, std


def export(model, mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    state = {key.replace(".", "_"): value.detach().double().numpy() for key, value in model.state_dict().items()}
    np.savez(path, mean=mean.astype(np.float64), std=std.astype(np.float64), **state)


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(prog="possession.external.train_temporal")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--label", choices=("y", "y1"), required=True)
    parser.add_argument("--fold", choices=MATCHES)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--with-randomized", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    tag = "temporal_e2" if args.label == "y" else "temporal_e1"
    if args.final:
        if args.model_dir is None:
            parser.error("--final needs --model-dir")
        model, mean, std = fit(torch, args.cache_dir, list(MATCHES), args.label, args.with_randomized)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        export(model, mean, std, args.model_dir / f"{tag}.npz")
    elif args.fold:
        from possession.temporal import dead_and_away

        model, mean, std = fit(torch, args.cache_dir, [m for m in MATCHES if m != args.fold], args.label, args.with_randomized)
        exported = args.cache_dir / "oof" / f"{tag}_{args.fold}.npz"
        exported.parent.mkdir(parents=True, exist_ok=True)
        export(model, mean, std, exported)
        with np.load(exported) as data:
            weights = {key: data[key].astype(np.float64) for key in data.files}
        for variant in VARIANTS:
            name = f"{args.fold}_v{variant:02d}"
            x, y = sequence(args.cache_dir, name, False, "y")
            out = np.full((y.size, 2), np.nan, dtype=np.float32)
            graded = y >= 0
            out[graded] = dead_and_away(weights, x)[graded]
            np.save(args.cache_dir / "oof" / f"{tag}_{name}.npy", out)
    else:
        parser.error("give --fold or --final")
    print(json.dumps({"label": args.label, "fold": args.fold, "final": args.final, "seconds": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
