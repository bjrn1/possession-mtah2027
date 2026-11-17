import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from possession.codes import D, H, X
from possession.external.dfl import ExternalMatch, match_files, read_match
from possession.external.labels import event_labels, native_labels
from possession.scoring import evaluate, headline


def build_match(idsse_dir: Path, synced_dir: Path, cache_dir: Path, match_id: str, force: bool) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = cache_dir / f"{match_id}_tracking.npz"
    labels_path = cache_dir / f"{match_id}_labels.npz"
    started = time.time()
    if tracking_path.is_file() and not force:
        match = ExternalMatch.load(tracking_path)
    else:
        info, _, positions = match_files(idsse_dir / match_id)
        match = read_match(info, positions)
        match.save(tracking_path)

    events = pd.read_parquet(synced_dir / f"{match_id}.parquet")
    native = native_labels(match)
    derived, report = event_labels(match, events)
    np.savez_compressed(labels_path, native=native, derived=derived, played=match.half > 0)

    agreement = headline(evaluate(derived, native))
    played = match.half > 0
    return {
        "match_id": match_id,
        "seconds": round(time.time() - started, 1),
        "frames": int(played.sum()),
        "players": int(match.slot_team.size),
        "ball_missing_share": round(float(np.isnan(match.ball[played, 0]).mean()), 4),
        "dead_share": round(float(np.mean(derived[played] == D)), 4),
        "events": report.n_events,
        "events_unplaced": report.n_unplaced,
        "control_events": report.n_control,
        "unknown_event_types": list(report.unknown_types),
        "alive_runs_without_control": report.runs_without_control,
        "derived_x_frames": int(np.sum(derived[played] == X)),
        "native_vs_derived": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agreement.items()},
        "home_share_derived": round(float(np.mean(derived[derived < D] == H)), 4) if np.any(derived < D) else None,
        "home_share_native": round(float(np.mean(native[native < D] == H)), 4) if np.any(native < D) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.build")
    parser.add_argument("--idsse-dir", required=True, type=Path)
    parser.add_argument("--synced-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--match", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    matches = args.match or sorted(p.name for p in args.idsse_dir.iterdir() if p.is_dir())
    for match_id in matches:
        print(json.dumps(build_match(args.idsse_dir, args.synced_dir, args.cache_dir, match_id, args.force)))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
