import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOME = 0
AWAY = 1
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","


@dataclass(frozen=True)
class TrackingMeta:
    team_ids: tuple[str, str]
    slot_team: np.ndarray

    @property
    def n_slots(self) -> int:
        return int(self.slot_team.shape[0])


@dataclass(frozen=True)
class Tracking:
    meta: TrackingMeta
    players: np.ndarray
    ball: np.ndarray
    present: np.ndarray
    pitch: tuple[float, float]

    @property
    def n_frames(self) -> int:
        return int(self.players.shape[0])

    @property
    def ball_seen(self) -> np.ndarray:
        return ~np.isnan(self.ball[:, 0])

    @property
    def player_seen(self) -> np.ndarray:
        return ~np.isnan(self.players[:, :, 0])

    def team_mask(self, team: int) -> np.ndarray:
        return self.meta.slot_team == team


def read_meta(path: str | Path) -> TrackingMeta:
    root = ET.parse(path).getroot()
    team_ids = tuple(team.get("id") for team in root.iter("Team"))
    if len(team_ids) != 2:
        raise ValueError(f"{path}: expected 2 teams, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{path}: no Player entries")
    unknown = sorted(set(slots) - set(team_ids))
    if unknown:
        raise ValueError(f"{path}: players reference unknown teams {unknown}")
    slot_team = np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)
    return TrackingMeta(team_ids=team_ids, slot_team=slot_team)


def _group_values(group: str, width: int) -> list[float]:
    parts = group.split(VALUE_SEP)
    if len(parts) < width:
        return [float("nan")] * width
    return [float(v) for v in parts[:width]]


def read_tracking(
    meta_path: str | Path,
    raw_path: str | Path,
    n_frames: int,
    pitch: tuple[float, float],
) -> Tracking:
    meta = read_meta(meta_path)
    length, width = pitch
    players = np.full((n_frames, meta.n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)

    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split(FIELD_SEP)
            if len(fields) != 3:
                raise ValueError(f"{raw_path}:{line_number}: expected 3 fields, got {len(fields)}")
            index = int(fields[0]) - 1
            if not 0 <= index < n_frames:
                raise ValueError(f"{raw_path}:{line_number}: frame id outside 1..{n_frames}")
            groups = fields[1].split(GROUP_SEP)
            if len(groups) != meta.n_slots:
                raise ValueError(f"{raw_path}:{line_number}: {len(groups)} slots, expected {meta.n_slots}")
            players[index] = [_group_values(g, 3) for g in groups]
            ball[index] = _group_values(fields[2], 2)
            present[index] = True

    players[:, :, 0] *= length
    players[:, :, 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return Tracking(meta=meta, players=players, ball=ball, present=present, pitch=pitch)


def scan_raw(raw_path: str | Path) -> dict:
    count = 0
    first = last = None
    contiguous = True
    slot_counts: set[int] = set()
    ball_widths: set[int] = set()
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split(FIELD_SEP)
            frame_id = int(fields[0])
            if first is None:
                first = frame_id
            elif frame_id != last + 1:
                contiguous = False
            last = frame_id
            count += 1
            if len(fields) == 3:
                slot_counts.add(len(fields[1].split(GROUP_SEP)))
                ball_widths.add(len(fields[2].split(VALUE_SEP)))
    return {
        "records": count,
        "first_id": first,
        "last_id": last,
        "contiguous": contiguous,
        "slots_per_line": sorted(slot_counts),
        "ball_fields": sorted(ball_widths),
    }
