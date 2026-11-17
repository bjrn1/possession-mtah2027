import argparse
import json
import sys
from pathlib import Path

import numpy as np

from possession.baseline import baseline_labels
from possession.external.degrade import load_tracking
from possession.scoring import evaluate, headline

METHODS = {
    "baseline_raw": 0,
    "baseline_smooth25": 25,
    "baseline_smooth50": 50,
    "baseline_smooth125": 125,
}


def variant_files(cache_dir: Path, matches: list[str] | None) -> list[Path]:
    files = sorted((cache_dir / "degraded").glob("*_v*.npz"))
    if matches:
        files = [f for f in files if f.name.split("_v")[0] in matches]
    return files


def score_variant(cache_dir: Path, path: Path) -> list[dict]:
    match_id = path.name.split("_v")[0]
    variant = int(path.stem.split("_v")[1])
    tracking = load_tracking(path)
    with np.load(cache_dir / f"{match_id}_labels.npz") as labels:
        references = {"E2": labels["derived"], "E1": labels["native"]}
    played = tracking.present
    ball_missing = float(np.isnan(tracking.ball[played, 0]).mean())
    rows = []
    for method, half_width in METHODS.items():
        prediction = baseline_labels(tracking, half_width, played)
        for name, reference in references.items():
            result = headline(evaluate(reference, prediction))
            rows.append(
                {
                    "match_id": match_id,
                    "variant": variant,
                    "ball_missing": round(ball_missing, 3),
                    "method": method,
                    "reference": name,
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()},
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.evaluate")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--match", nargs="+")
    args = parser.parse_args(argv)
    for path in variant_files(args.cache_dir, args.match):
        for row in score_variant(args.cache_dir, path):
            print(json.dumps(row))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
