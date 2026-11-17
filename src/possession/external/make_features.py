import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from possession.codes import A, D, H, X
from possession.external.degrade import load_tracking
from possession.external.dfl import ExternalMatch
from possession.external.randomize import random_settings, randomized
from possession.features import frame_features, swap_teams
from possession.features_control import NAMES as CONTROL_NAMES
from possession.features_control import control_features

SWAPPED_LABEL = np.array([A, H, D, X], dtype=np.int8)
LABEL_SETS = {"y": "derived", "y1": "native"}


def feature_path(cache_dir: Path, name: str, swapped: bool, part: str) -> Path:
    suffix = "_swap" if swapped else ""
    return cache_dir / "features" / f"{name}{suffix}_{part}.npy"


MATCH_ORDER = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def make(cache_dir: Path, name: str, with_randomized: bool = False) -> dict:
    started = time.time()
    tracking = load_tracking(cache_dir / "degraded" / f"{name}.npz")
    match_id = name.split("_v")[0]
    with np.load(cache_dir / f"{match_id}_labels.npz") as labels:
        targets = {part: labels[key].astype(np.int8) for part, key in LABEL_SETS.items()}
    if with_randomized:
        seed = 7000 + MATCH_ORDER.index(match_id) * 10 + int(name.split("_v")[1])
        clean = ExternalMatch.load(cache_dir / f"{match_id}_tracking.npz").ball.astype(np.float64)
        tracking = randomized(tracking, clean, targets["y"], seed=seed, **random_settings(seed))
        name = f"dr_{name}"
    (cache_dir / "features").mkdir(parents=True, exist_ok=True)
    names = None
    for swapped, source in ((False, tracking), (True, swap_teams(tracking))):
        base = frame_features(source)
        names = list(base.names) + list(CONTROL_NAMES)
        np.save(feature_path(cache_dir, name, swapped, "x"), base.values.astype(np.float16))
        np.save(feature_path(cache_dir, name, swapped, "c"), control_features(source).astype(np.float16))
        for part, target in targets.items():
            np.save(feature_path(cache_dir, name, swapped, part), SWAPPED_LABEL[target] if swapped else target)
    (cache_dir / "features" / "names.json").write_text(json.dumps(names))
    return {"name": name, "features": len(names), "seconds": round(time.time() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.make_features")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--randomized", action="store_true")
    args = parser.parse_args(argv)
    for name in args.names:
        print(json.dumps(make(args.cache_dir, name, args.randomized)))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
