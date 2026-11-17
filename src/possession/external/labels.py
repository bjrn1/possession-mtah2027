from dataclasses import dataclass

import numpy as np
import pandas as pd

from possession.codes import A, D, H, X, runs
from possession.external.dfl import ExternalMatch

CONTROL_ACTIONS = frozenset(
    {
        "control",
        "pass",
        "cross",
        "shot",
        "shot_freekick",
        "shot_penalty",
        "dribble",
        "take_on",
        "throw_in",
        "freekick_short",
        "freekick_crossed",
        "corner_short",
        "corner_crossed",
        "goalkick",
        "ball_recovery",
        "interception",
        "keeper_pick_up",
        "keeper_claim",
    }
)
NON_CONTROL_ACTIONS = frozenset(
    {
        "bad_touch",
        "clearance",
        "shot_block",
        "tackle",
        "foul",
        "dispossessed",
        "keeper_punch",
        "keeper_save",
        "out",
        "goal",
    }
)
TEAM_PREFIX = {"home": H, "away": A}


@dataclass(frozen=True)
class EventLabelReport:
    n_events: int
    n_unplaced: int
    n_control: int
    unknown_types: tuple[str, ...]
    runs_without_control: int


RESTART_LOOKBACK = 25


def _carry_forward(codes: np.ndarray, alive: np.ndarray) -> np.ndarray:
    out = codes.copy()
    for start, stop in runs(alive):
        window = out[start:stop]
        known = np.flatnonzero(window != X)
        if known.size == 0:
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(window.size), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = window[carry]
    return out


def native_labels(match: ExternalMatch) -> np.ndarray:
    codes = np.full(match.n_frames, X, dtype=np.int8)
    played = match.half > 0
    alive = played & (match.ball_status == 1)
    codes[played & ~alive] = D
    codes[alive & (match.ball_possession == 1)] = H
    codes[alive & (match.ball_possession == 2)] = A
    return _carry_forward(codes, alive)


def _event_index(match: ExternalMatch, events: pd.DataFrame) -> np.ndarray:
    lookup = {}
    for half in (1, 2):
        frames = np.flatnonzero(match.half == half)
        lookup[half] = (int(match.frame_n[frames[0]]), int(frames[0]), int(frames.size))
    out = np.full(len(events), -1, dtype=np.int64)
    for position, (period, frame_id) in enumerate(zip(events["period_id"].to_numpy(), events["frame_id"].to_numpy())):
        if pd.isna(frame_id) or int(period) not in lookup:
            continue
        first_n, first_index, size = lookup[int(period)]
        offset = int(frame_id) - first_n
        if 0 <= offset < size:
            out[position] = first_index + offset
    return out


def event_labels(match: ExternalMatch, events: pd.DataFrame) -> tuple[np.ndarray, EventLabelReport]:
    native = native_labels(match)
    codes = np.full(match.n_frames, X, dtype=np.int8)
    played = match.half > 0
    alive = played & (match.ball_status == 1)
    codes[played & ~alive] = D

    index = _event_index(match, events)
    team = events["player_id"].astype(str).str.split("_").str[0].map(TEAM_PREFIX).to_numpy()
    kind = events["spadl_type"].astype(str).to_numpy()
    is_control = np.isin(kind, list(CONTROL_ACTIONS)) & ~pd.isna(team)
    unknown = sorted(set(kind) - CONTROL_ACTIONS - NON_CONTROL_ACTIONS)

    placed = (index >= 0) & is_control
    order = np.argsort(index[placed], kind="stable")
    control_frames = index[placed][order]
    control_teams = team[placed][order].astype(np.int8)

    without_control = 0
    for start, stop in runs(alive):
        left = np.searchsorted(control_frames, max(0, start - RESTART_LOOKBACK), side="left")
        right = np.searchsorted(control_frames, stop, side="left")
        frames = control_frames[left:right]
        teams = control_teams[left:right]
        if frames.size == 0:
            without_control += 1
            fallback = native[start]
            codes[start:stop] = fallback if fallback in (H, A) else X
            continue
        before = frames < start
        current = int(teams[before][-1]) if before.any() else int(teams[0])
        frames = frames[~before]
        teams = teams[~before]
        cursor = start
        for frame, owner in zip(frames.tolist(), teams.tolist()):
            if owner != current:
                codes[cursor:frame] = current
                cursor = frame
                current = int(owner)
        codes[cursor:stop] = current

    report = EventLabelReport(
        n_events=len(events),
        n_unplaced=int(np.sum(index < 0)),
        n_control=int(placed.sum()),
        unknown_types=tuple(unknown),
        runs_without_control=without_control,
    )
    return codes, report
