import os
from pathlib import Path

import pytest

from possession.external.check_direction import check_match

IDSSE = os.environ.get("MTAH_IDSSE_DIR")


@pytest.mark.skipif(IDSSE is None, reason="MTAH_IDSSE_DIR is not set")
def test_detected_attack_direction_matches_dfl_kickoff_metadata():
    folders = sorted(path for path in Path(IDSSE).iterdir() if path.is_dir())
    assert len(folders) == 7
    rows = [row for folder in folders for row in check_match(folder)]
    assert len(rows) == 14
    assert all(expected == detected for _, _, expected, detected in rows)
