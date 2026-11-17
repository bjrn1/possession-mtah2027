import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from possession.codes import D, X, to_letters
from possession.decoder import DecoderParams, decode
from possession.features import frame_features
from possession.features_control import control_features
from possession.io_atd import read_tracking
from possession.kickoff import dead_until_kickoff
from possession.paths import HALF_VIDEO_BLOCKS, PITCH_METRES, bundle
from possession.temporal import dead_and_away, load_network

REPOSITORY = Path(__file__).resolve().parents[2]
EPSILON = 1e-4
CHUNK = 50000


def graded_mask(n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in HALF_VIDEO_BLOCKS:
        mask[first:last + 1] = True
    return mask


def graded_blocks() -> list[tuple[int, int]]:
    return [(first, last + 1) for _, first, last in HALF_VIDEO_BLOCKS]


def verify_models(config: dict) -> None:
    for relative, expected in config.get("model_sha256", {}).items():
        actual = hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{relative}: hash {actual} does not match the frozen configuration")


def model_inputs(tracking) -> np.ndarray:
    values = np.concatenate([frame_features(tracking).values, control_features(tracking)], axis=1)
    return values.astype(np.float16).astype(np.float32)


def member_probabilities(member: dict, features: np.ndarray, played: np.ndarray) -> np.ndarray:
    out = np.full((features.shape[0], 2), np.nan)
    rows = np.flatnonzero(played)
    if member["kind"] == "trees":
        dead_model = joblib.load(REPOSITORY / member["dead"])
        team_model = joblib.load(REPOSITORY / member["team"])
        for start in range(0, rows.size, CHUNK):
            chunk = rows[start:start + CHUNK]
            out[chunk, 0] = dead_model.predict_proba(features[chunk])[:, 1]
            out[chunk, 1] = team_model.predict_proba(features[chunk])[:, 1]
    elif member["kind"] == "temporal":
        probabilities = dead_and_away(load_network(REPOSITORY / member["network"]), features)
        out[rows] = probabilities[rows]
    else:
        raise ValueError(f"unknown member kind {member['kind']}")
    return np.round(out, 4)


def combine(members: list[np.ndarray]) -> np.ndarray:
    logits = []
    for probabilities in members:
        clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
        logits.append(np.log(clipped) - np.log1p(-clipped))
    mean = np.mean(logits, axis=0)
    return np.round(1.0 / (1.0 + np.exp(-mean)), 4)


def write_submission(path: Path, labels: np.ndarray) -> None:
    letters = to_letters(labels)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{letter}\n" for frame, letter in enumerate(letters))


def without_unlabelled(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    for _, first, last in HALF_VIDEO_BLOCKS:
        block = out[first:last + 1]
        known = np.flatnonzero(block != X)
        if known.size == 0:
            block[:] = D
            continue
        index = np.arange(block.size)
        previous = np.maximum.accumulate(np.where(block != X, index, -1))
        previous = np.where(previous < 0, known[0], previous)
        block[:] = block[previous]
    out[out == X] = D
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.run")
    parser.add_argument("--data-dir")
    parser.add_argument("--config", default=str(REPOSITORY / "config" / "frozen.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    verify_models(config)
    source = bundle(args.data_dir, "tactical")
    tracking = read_tracking(source.path("atd_meta"), source.path("atd_raw"), source.n_frames, PITCH_METRES)
    played = graded_mask(source.n_frames) & tracking.present
    tracking = type(tracking)(meta=tracking.meta, players=tracking.players, ball=tracking.ball, present=played, pitch=tracking.pitch)
    features = model_inputs(tracking)
    probabilities = combine([member_probabilities(member, features, played) for member in config["members"]])
    labels = decode(probabilities, played, DecoderParams(**config["decoder"]))
    if config.get("dead_until_kickoff", False):
        labels = dead_until_kickoff(labels, tracking, probabilities[:, 0], graded_blocks())
    labels = without_unlabelled(labels)
    out = Path(args.out)
    write_submission(out, labels)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
