import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession.codes import A, D, X
from possession.external.make_features import feature_path

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
VARIANTS = range(5)
STRIDE = 25
MIN_WEIGHT_FRAMES = 25
CHUNK = 50000


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def possession_weights(labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(labels.size, dtype=np.float32)
    for code in (0, 1):
        padded = np.concatenate(([0], (labels == code).astype(np.int8), [0]))
        edges = np.diff(padded)
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            weights[start:stop] = 1.0 / max(stop - start, MIN_WEIGHT_FRAMES)
    return weights


def inputs(cache_dir: Path, name: str, swapped: bool, rows: np.ndarray) -> np.ndarray:
    base = np.load(feature_path(cache_dir, name, swapped, "x"), mmap_mode="r")
    control = np.load(feature_path(cache_dir, name, swapped, "c"), mmap_mode="r")
    return np.concatenate([np.asarray(base[rows], dtype=np.float32), np.asarray(control[rows], dtype=np.float32)], axis=1)


RANDOMIZED_SHIFT = 13


def training_rows(cache_dir: Path, matches: list[str], label: str, with_randomized: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for match_index, match_id in enumerate(matches):
        for variant in VARIANTS:
            for swapped in (False, True):
                base_name = f"{match_id}_v{variant:02d}"
                sources = [(base_name, 0)] + ([(f"dr_{base_name}", RANDOMIZED_SHIFT)] if with_randomized else [])
                for name, shift in sources:
                    y = np.load(feature_path(cache_dir, name, swapped, label))
                    offset = (match_index * 7 + variant * 3 + int(swapped) + shift) % STRIDE
                    rows = np.flatnonzero(y != X)
                    rows = rows[(rows % STRIDE) == offset]
                    xs.append(inputs(cache_dir, name, swapped, rows))
                    ys.append(y[rows])
                    ws.append(possession_weights(y)[rows])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def fit(cache_dir: Path, matches: list[str], label: str, with_dead: bool, with_randomized: bool) -> tuple:
    x, y, w = training_rows(cache_dir, matches, label, with_randomized)
    dead = make_model(0).fit(x, (y == D).astype(np.int8)) if with_dead else None
    alive = y != D
    team = make_model(3).fit(x[alive], (y[alive] == A).astype(np.int8), sample_weight=w[alive] / w[alive].mean())
    return dead, team


def predict(dead, team, cache_dir: Path, name: str) -> np.ndarray:
    labels = np.load(feature_path(cache_dir, name, False, "y"))
    out = np.full((labels.size, 2), np.nan, dtype=np.float32)
    rows = np.flatnonzero(labels != X)
    for start in range(0, rows.size, CHUNK):
        chunk = rows[start:start + CHUNK]
        values = inputs(cache_dir, name, False, chunk)
        out[chunk, 0] = dead.predict_proba(values)[:, 1]
        out[chunk, 1] = team.predict_proba(values)[:, 1]
    return out


def run_fold(cache_dir: Path, held_out: str, with_randomized: bool) -> dict:
    started = time.time()
    matches = [m for m in MATCHES if m != held_out]
    dead, team_e2 = fit(cache_dir, matches, "y", True, with_randomized)
    _, team_e1 = fit(cache_dir, matches, "y1", False, with_randomized)
    folder = cache_dir / "oof"
    folder.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        name = f"{held_out}_v{variant:02d}"
        np.save(folder / f"trees_e2_{name}.npy", predict(dead, team_e2, cache_dir, name))
        np.save(folder / f"trees_e1_{name}.npy", predict(dead, team_e1, cache_dir, name))
    return {"held_out": held_out, "seconds": round(time.time() - started, 1)}


def run_final(cache_dir: Path, model_dir: Path, with_randomized: bool) -> dict:
    started = time.time()
    dead, team_e2 = fit(cache_dir, list(MATCHES), "y", True, with_randomized)
    _, team_e1 = fit(cache_dir, list(MATCHES), "y1", False, with_randomized)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dead, model_dir / "trees_dead.joblib")
    joblib.dump(team_e2, model_dir / "trees_team_e2.joblib")
    joblib.dump(team_e1, model_dir / "trees_team_e1.joblib")
    return {"final": True, "seconds": round(time.time() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.train")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--fold", choices=MATCHES)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--with-randomized", action="store_true")
    args = parser.parse_args(argv)
    if args.final:
        if args.model_dir is None:
            parser.error("--final needs --model-dir")
        result = run_final(args.cache_dir, args.model_dir, args.with_randomized)
    elif args.fold:
        result = run_fold(args.cache_dir, args.fold, args.with_randomized)
    else:
        parser.error("give --fold or --final")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
