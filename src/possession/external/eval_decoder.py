import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from possession.decoder import DecoderParams, decode
from possession.external.train import MATCHES, VARIANTS
from possession.run import combine
from possession.vendor import possession_metrics as pm

MEMBERS = ("trees_e2", "trees_e1", "temporal_e2", "temporal_e1")


def quick_scores(reference: np.ndarray, prediction: np.ndarray) -> dict:
    graded = pm.valid_runs(reference)
    spells = pm.spell_scores(reference, prediction, graded)
    cm3 = pm.scored_confusion(pm.confusion(reference, prediction))
    return {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(cm3), "n_pred": spells["n_pred"], "n_gt": spells["n_gt"]}


def load_variant(cache_dir: Path, match_id: str, variant: int, members: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    name = f"{match_id}_v{variant:02d}"
    parts = [np.round(np.load(cache_dir / "oof" / f"{member}_{name}.npy").astype(np.float64), 4) for member in members]
    with np.load(cache_dir / f"{match_id}_labels.npz") as labels:
        references = {"E2": labels["derived"], "E1": labels["native"], "played": labels["played"]}
    return combine(parts), references


def grid(spec: dict) -> list[DecoderParams]:
    keys = list(spec)
    return [DecoderParams(**dict(zip(keys, values))) for values in itertools.product(*(spec[k] for k in keys))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.eval_decoder")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--members", nargs="+", default=list(MEMBERS))
    parser.add_argument("--references", nargs="+", default=["E2", "E1"])
    args = parser.parse_args(argv)
    configs = grid(json.loads(args.grid))
    members = tuple(args.members)
    data = {(m, v): load_variant(args.cache_dir, m, v, members) for m in MATCHES for v in VARIANTS}
    for params in configs:
        started = time.time()
        for (match_id, variant), (probabilities, references) in data.items():
            played = references["played"] & ~np.isnan(probabilities[:, 0])
            prediction = decode(probabilities, played, params)
            for name in args.references:
                row = {"match_id": match_id, "variant": variant, "reference": name, "members": "+".join(members), **asdict(params), **quick_scores(references[name], prediction)}
                print(json.dumps(row))
        sys.stdout.flush()
        print(json.dumps({"config_seconds": round(time.time() - started, 1)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
