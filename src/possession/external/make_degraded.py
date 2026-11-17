import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from possession.external.degrade import degrade, sample_params, save_tracking
from possession.external.dfl import ExternalMatch


MAX_VARIANTS = 100


def variant_seed(match_id: str, index: int) -> int:
    if not 0 <= index < MAX_VARIANTS:
        raise ValueError(f"variant index must be in 0..{MAX_VARIANTS - 1}")
    return (zlib.crc32(match_id.encode("ascii")) % 1_000_000) * MAX_VARIANTS + index


def make_variant(cache_dir: Path, match_id: str, index: int) -> dict:
    started = time.time()
    match = ExternalMatch.load(cache_dir / f"{match_id}_tracking.npz")
    params = sample_params(variant_seed(match_id, index))
    tracking = degrade(match, params)
    out_dir = cache_dir / "degraded"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tracking(tracking, params, out_dir / f"{match_id}_v{index:02d}.npz")
    played = match.half > 0
    tracked = np.sum(~np.isnan(tracking.players[played, :, 0]), axis=1)
    return {
        "match_id": match_id,
        "variant": index,
        "seed": params.seed,
        "seconds": round(time.time() - started, 1),
        "target_ball_missing": round(params.ball_missing, 3),
        "ball_missing": round(float(np.isnan(tracking.ball[played, 0]).mean()), 3),
        "slots": int(tracking.meta.slot_team.size),
        "mean_tracked": round(float(tracked.mean()), 2),
        "max_tracked": int(tracked.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.make_degraded")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--match", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    for match_id in args.match:
        for index in args.variants:
            print(json.dumps(make_variant(args.cache_dir, match_id, index)))
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
