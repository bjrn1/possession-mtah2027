import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from possession.external.dfl import HALVES, match_files, read_match, read_match_info
from possession.features import home_attacks_right


def expected_direction(events_path: Path, home_id: str) -> dict[str, bool]:
    left_team: dict[str, str] = {}
    for kickoff in ET.parse(events_path).getroot().iter("KickOff"):
        section = kickoff.get("GameSection")
        if section is not None and kickoff.get("TeamLeft") is not None:
            left_team.setdefault(section, kickoff.get("TeamLeft"))
    return {section: left_team[section] == home_id for section in HALVES}


def check_match(folder: Path) -> list[tuple[str, str, bool, bool]]:
    info_path, events_path, positions_path = match_files(folder)
    info = read_match_info(info_path)
    match = read_match(info_path, positions_path)
    expected = expected_direction(events_path, info.home_id)
    tracking = SimpleNamespace(players=match.players, meta=SimpleNamespace(slot_team=match.slot_team))
    rows = []
    for half_index, section in enumerate(HALVES, start=1):
        frames = np.flatnonzero(match.half == half_index)
        detected = home_attacks_right(tracking, int(frames[0]), int(frames[-1]) + 1)
        rows.append((info.match_id, section, expected[section], detected))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="possession.external.check_direction")
    parser.add_argument("--idsse-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    failures = 0
    for folder in sorted(path for path in args.idsse_dir.iterdir() if path.is_dir()):
        for match_id, section, expected, detected in check_match(folder):
            agree = expected == detected
            failures += not agree
            print(f"{match_id} {section} expected {expected} detected {detected} {'OK' if agree else 'FAIL'}")
    print("OK" if failures == 0 else f"FAIL {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
