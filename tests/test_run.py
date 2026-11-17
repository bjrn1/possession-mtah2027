import ast
import io
import json
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

from possession.codes import A, D, H, X
from possession.paths import HALF_VIDEO_BLOCKS
from possession.run import REPOSITORY, combine, verify_models, without_unlabelled

SOURCE = REPOSITORY / "src" / "possession"
CONFIG = REPOSITORY / "config" / "frozen.json"


def test_without_unlabelled_removes_every_x():
    labels = np.full(199319, X, dtype=np.int8)
    first, last = HALF_VIDEO_BLOCKS[0][1], HALF_VIDEO_BLOCKS[0][2]
    labels[first + 10:first + 100] = H
    labels[first + 200:first + 300] = A
    out = without_unlabelled(labels)
    assert not (out == X).any()
    assert out[first] == H
    assert out[first + 150] == H
    assert out[first + 350] == A
    assert out[0] == D


def test_without_unlabelled_keeps_known_labels():
    labels = np.full(199319, D, dtype=np.int8)
    first = HALF_VIDEO_BLOCKS[1][1]
    labels[first:first + 500] = A
    out = without_unlabelled(labels)
    assert (out[first:first + 500] == A).all()


def test_source_contains_no_match_specific_frame_constants():
    forbidden = re.compile(r"\b(113742|113_742|2934|158857|94214|113488)\b")
    offenders = [path.name for path in SOURCE.rglob("*.py") if forbidden.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_half_blocks_are_only_used_to_separate_halves():
    users = sorted(path.name for path in SOURCE.rglob("*.py") if "HALF_VIDEO_BLOCKS" in path.read_text(encoding="utf-8"))
    assert users == ["paths.py", "run.py"]


def test_source_has_no_forbidden_characters_or_comments():
    characters = {chr(code) for code in (0x2014, 0x2013, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x2192)}
    own = [path for path in list(SOURCE.rglob("*.py")) + list((REPOSITORY / "tests").rglob("*.py"))]
    for path in own:
        text = path.read_text(encoding="utf-8")
        assert not characters & set(text), path.name
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        assert not [token for token in tokens if token.type == tokenize.COMMENT], path.name
        tree = ast.parse(text)
        documented = [node for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node)]
        assert not documented, path.name


def test_frozen_models_match_their_hashes():
    verify_models(json.loads(CONFIG.read_text()))


def test_frozen_configuration_lists_four_members_and_their_files():
    config = json.loads(CONFIG.read_text())
    assert config["decoder"]["boundary_weight"] == 0.0
    assert [member["kind"] for member in config["members"]] == ["trees", "trees", "temporal", "temporal"]
    files = {path for member in config["members"] for key, path in member.items() if key != "kind"}
    assert files == set(config["model_sha256"])


def test_combine_averages_member_logits():
    first = np.array([[0.9, 0.2], [0.5, 0.5]])
    second = np.array([[0.1, 0.8], [0.5, 0.5]])
    combined = combine([first, second])
    assert np.allclose(combined, 0.5)


def test_verify_models_rejects_a_changed_hash(tmp_path):
    config = json.loads(CONFIG.read_text())
    first = next(iter(config["model_sha256"]))
    config["model_sha256"][first] = "0" * 64
    with pytest.raises(ValueError):
        verify_models(config)
