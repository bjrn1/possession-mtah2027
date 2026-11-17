import argparse
import json
import sys
from pathlib import Path

import pandas as pd

KEYS = (
    "switch_penalty",
    "stop_penalty",
    "restart_penalty",
    "min_possession_frames",
    "min_dead_frames",
    "dead_bias",
    "emission_weight",
    "boundary_weight",
)


def load(paths: list[Path], reference: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)
    frame = frame[frame["reference"] == reference].copy()
    frame["config"] = frame[list(KEYS)].astype(str).agg("|".join, axis=1)
    return frame


def nested(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id in sorted(frame["match_id"].unique()):
        training = frame[frame["match_id"] != match_id].groupby("config")["spell_f1"].mean()
        chosen = training.idxmax()
        held = frame[(frame["match_id"] == match_id) & (frame["config"] == chosen)]
        rows.append({"match_id": match_id, "config": chosen, "spell_f1": held["spell_f1"].mean(), "macro_f1": held["macro_f1"].mean()})
    return pd.DataFrame(rows)


def final_choice(frame: pd.DataFrame) -> dict:
    means = frame.groupby("config")[["spell_f1", "macro_f1"]].mean()
    best = means["spell_f1"].idxmax()
    values = dict(zip(KEYS, best.split("|")))
    decoder = {k: (int(float(v)) if k.startswith("min_") else float(v)) for k, v in values.items()}
    return {"decoder": decoder, "spell_f1_all_variants": round(float(means.loc[best, "spell_f1"]), 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.select_decoder")
    parser.add_argument("--results", nargs="+", required=True, type=Path)
    parser.add_argument("--reference", default="E2")
    args = parser.parse_args(argv)
    frame = load(args.results, args.reference)
    table = nested(frame)
    print(json.dumps({
        "nested_leave_one_match_out_spell_f1": round(float(table["spell_f1"].mean()), 4),
        "nested_leave_one_match_out_macro_f1": round(float(table["macro_f1"].mean()), 4),
        "per_match": {r.match_id: round(float(r.spell_f1), 4) for r in table.itertuples()},
        **final_choice(frame),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
