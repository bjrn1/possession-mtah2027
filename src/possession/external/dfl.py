import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.io_atd import AWAY, HOME

BALL_TEAM = "BALL"
HALVES = ("firstHalf", "secondHalf")
FRAMESET = re.compile(r'<FrameSet\s[^>]*?GameSection="([^"]+)"[^>]*?TeamId="([^"]+)"[^>]*?PersonId="([^"]+)"')
ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')
KMH_TO_MS = 1.0 / 3.6


@dataclass(frozen=True)
class MatchInfo:
    match_id: str
    home_id: str
    away_id: str
    pitch: tuple[float, float]


@dataclass(frozen=True)
class ExternalMatch:
    match_id: str
    pitch: tuple[float, float]
    players: np.ndarray
    slot_team: np.ndarray
    ball: np.ndarray
    ball_z: np.ndarray
    half: np.ndarray
    frame_n: np.ndarray
    ball_status: np.ndarray
    ball_possession: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.half.shape[0])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            match_id=self.match_id,
            pitch=np.array(self.pitch),
            players=self.players,
            slot_team=self.slot_team,
            ball=self.ball,
            ball_z=self.ball_z,
            half=self.half,
            frame_n=self.frame_n,
            ball_status=self.ball_status,
            ball_possession=self.ball_possession,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExternalMatch":
        with np.load(path) as data:
            return cls(
                match_id=str(data["match_id"]),
                pitch=tuple(float(v) for v in data["pitch"]),
                players=data["players"],
                slot_team=data["slot_team"],
                ball=data["ball"],
                ball_z=data["ball_z"],
                half=data["half"],
                frame_n=data["frame_n"],
                ball_status=data["ball_status"],
                ball_possession=data["ball_possession"],
            )


def read_match_info(path: str | Path) -> MatchInfo:
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    roles = {team.get("Role"): team.get("TeamId") for team in root.iter("Team")}
    if "home" not in roles or "guest" not in roles:
        raise ValueError(f"{path}: home or guest team missing")
    return MatchInfo(
        match_id=general.get("MatchId"),
        home_id=roles["home"],
        away_id=roles["guest"],
        pitch=(float(environment.get("PitchX")), float(environment.get("PitchY"))),
    )


def _read_framesets(path: str | Path) -> list[dict]:
    framesets = []
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "<Frame " in line or "<Frame\t" in line:
                if current is None:
                    continue
                attributes = dict(ATTRIBUTE.findall(line))
                try:
                    n = int(attributes["N"])
                    point = (float(attributes["X"]), float(attributes["Y"]), float(attributes["S"]) * KMH_TO_MS)
                    codes = (int(attributes["BallPossession"]), int(attributes["BallStatus"])) if current["ball"] else None
                    z = float(attributes.get("Z", "nan"))
                except (KeyError, ValueError) as error:
                    raise ValueError(f"{path}: unreadable Frame line: {line.strip()[:120]}") from error
                current["n"].append(n)
                current["xys"].append(point)
                if current["ball"]:
                    current["z"].append(z)
                    current["codes"].append(codes)
            elif "<FrameSet" in line:
                header = FRAMESET.search(line)
                if header is None:
                    raise ValueError(f"{path}: unreadable FrameSet header")
                section, team, person = header.groups()
                current = {
                    "section": section,
                    "team": team,
                    "person": person,
                    "ball": team == BALL_TEAM,
                    "n": [],
                    "xys": [],
                    "z": [],
                    "codes": [],
                }
            elif "</FrameSet>" in line and current is not None:
                current["n"] = np.asarray(current["n"], dtype=np.int64)
                current["xys"] = np.asarray(current["xys"], dtype=np.float32).reshape(-1, 3)
                current["z"] = np.asarray(current["z"], dtype=np.float32)
                current["codes"] = np.asarray(current["codes"], dtype=np.int8).reshape(-1, 2)
                framesets.append(current)
                current = None
    return framesets


def _to_corner_origin(xy: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    length, width = pitch
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] + length / 2.0
    out[..., 1] = width / 2.0 - xy[..., 1]
    return out


def read_match(info_path: str | Path, positions_path: str | Path) -> ExternalMatch:
    info = read_match_info(info_path)
    framesets = _read_framesets(positions_path)
    team_of = {info.home_id: HOME, info.away_id: AWAY}
    persons = sorted({(team_of[f["team"]], f["person"]) for f in framesets if f["team"] in team_of})
    slot_of = {person: index for index, (_, person) in enumerate(persons)}
    slot_team = np.array([team for team, _ in persons], dtype=np.int8)

    blocks = []
    for half_index, section in enumerate(HALVES, start=1):
        balls = [f for f in framesets if f["ball"] and f["section"] == section]
        if len(balls) != 1:
            raise ValueError(f"{positions_path}: expected one ball FrameSet in {section}, found {len(balls)}")
        ball = balls[0]
        first, last = int(ball["n"].min()), int(ball["n"].max())
        size = last - first + 1
        players = np.full((size, len(persons), 3), np.nan, dtype=np.float32)
        for frameset in framesets:
            if frameset["section"] != section or frameset["team"] not in team_of:
                continue
            index = frameset["n"] - first
            keep = (index >= 0) & (index < size)
            players[index[keep], slot_of[frameset["person"]]] = frameset["xys"][keep]
        ball_xy = np.full((size, 2), np.nan, dtype=np.float32)
        ball_z = np.full(size, np.nan, dtype=np.float32)
        status = np.zeros(size, dtype=np.int8)
        possession = np.zeros(size, dtype=np.int8)
        index = ball["n"] - first
        ball_xy[index] = ball["xys"][:, :2]
        ball_z[index] = ball["z"]
        possession[index] = ball["codes"][:, 0]
        status[index] = ball["codes"][:, 1]
        players[:, :, :2] = _to_corner_origin(players[:, :, :2], info.pitch)
        ball_xy = _to_corner_origin(ball_xy, info.pitch)
        blocks.append(
            {
                "players": players,
                "ball": ball_xy,
                "ball_z": ball_z,
                "half": np.full(size, half_index, dtype=np.int8),
                "frame_n": np.arange(first, last + 1, dtype=np.int64),
                "ball_status": status,
                "ball_possession": possession,
            }
        )

    separator = {
        "players": np.full((1, len(persons), 3), np.nan, dtype=np.float32),
        "ball": np.full((1, 2), np.nan, dtype=np.float32),
        "ball_z": np.full(1, np.nan, dtype=np.float32),
        "half": np.zeros(1, dtype=np.int8),
        "frame_n": np.full(1, -1, dtype=np.int64),
        "ball_status": np.zeros(1, dtype=np.int8),
        "ball_possession": np.zeros(1, dtype=np.int8),
    }
    ordered = [blocks[0], separator, blocks[1]]
    return ExternalMatch(
        match_id=info.match_id,
        pitch=info.pitch,
        slot_team=slot_team,
        **{key: np.concatenate([block[key] for block in ordered]) for key in separator},
    )


def match_files(folder: str | Path) -> tuple[Path, Path, Path]:
    folder = Path(folder)
    info = sorted(folder.glob("DFL_02_01_matchinformation_*.xml"))
    events = sorted(folder.glob("DFL_03_02_events_raw_*.xml"))
    positions = sorted(folder.glob("DFL_04_03_positions_raw_observed_*.xml"))
    if len(info) != 1 or len(events) != 1 or len(positions) != 1:
        raise FileNotFoundError(f"{folder}: expected one file of each DFL type")
    return info[0], events[0], positions[0]
