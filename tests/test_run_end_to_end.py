import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from possession.paths import GAME, N_FRAMES
from possession.run import REPOSITORY, main

pytestmark = pytest.mark.skipif(os.environ.get("MTAH_E2E") != "1", reason="MTAH_E2E is not set")


def write_synthetic_bundle(root: Path) -> None:
    folder = root / "tactical"
    folder.mkdir(parents=True)
    players = "".join(f'<Player id="P{i}" teamId="T{1 + i % 2}"/>' for i in range(46))
    (folder / f"{GAME}_tactical_atd.xml").write_text(
        '<?xml version="1.0"?><main><Metadata><Teams><Team id="T1"><Name>Home</Name></Team>'
        f'<Team id="T2"><Name>Away</Name></Team></Teams><Players>{players}</Players></Metadata></main>',
        encoding="utf-8",
    )
    rng = np.random.default_rng(7)
    n = N_FRAMES["tactical"]
    steps = rng.normal(0.0, 0.0006, size=(n, 22, 2)).cumsum(axis=0)
    base = np.concatenate([np.stack([rng.uniform(0.05, 0.45, 11), rng.uniform(0.1, 0.9, 11)], axis=1), np.stack([rng.uniform(0.55, 0.95, 11), rng.uniform(0.1, 0.9, 11)], axis=1)])
    positions = np.clip(base[None] + 0.2 * np.tanh(steps / 0.2), 0.0, 1.0)
    ball = np.clip(0.5 + 0.3 * np.tanh(rng.normal(0.0, 0.002, size=(n, 2)).cumsum(axis=0) / 0.3), 0.0, 1.0)
    ball_missing = rng.random(n) < 0.4
    lines = []
    for frame in range(n):
        cells = [f"{positions[frame, s, 0]:.4f},{positions[frame, s, 1]:.4f},1.0" for s in range(22)]
        cells += ["NaN,NaN,NaN"] * 24
        ball_cell = "NaN,NaN,NaN" if ball_missing[frame] else f"{ball[frame, 0]:.4f},{ball[frame, 1]:.4f},0.0"
        lines.append(f"{frame + 1}:" + ";".join(cells) + ":" + ball_cell)
    (folder / f"{GAME}_tactical_atd.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_single_run_produces_a_valid_deterministic_file(tmp_path):
    data = tmp_path / "data"
    write_synthetic_bundle(data)
    outputs = []
    for index in range(2):
        out = tmp_path / f"run{index}.csv"
        assert main(["--data-dir", str(data), "--out", str(out)]) == 0
        outputs.append(out)
    first = outputs[0].read_bytes()
    assert first == outputs[1].read_bytes()
    rows = first.decode("utf-8").splitlines()
    assert rows[0] == "frame,label"
    assert len(rows) == N_FRAMES["tactical"] + 1
    assert [int(row.split(",")[0]) for row in rows[1:]] == list(range(N_FRAMES["tactical"]))
    assert {row.split(",")[1] for row in rows[1:]} <= {"H", "A", "D"}
    sidecar = outputs[0].with_name(outputs[0].name + ".sha256").read_text().strip()
    assert sidecar == hashlib.sha256(first).hexdigest()
