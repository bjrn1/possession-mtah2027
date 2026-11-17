import numpy as np
import pandas as pd
import pytest

from possession.codes import A, D, H, X
from possession.external.dfl import read_match
from possession.external.labels import event_labels, native_labels
from possession.scoring import evaluate

INFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-TEST01"/>
<Environment PitchX="100.00" PitchY="60.00"/>
<Teams>
<Team TeamId="CLU-H" Role="home"/>
<Team TeamId="CLU-A" Role="guest"/>
</Teams>
</MatchInformation></PutDataRequest>
"""


def _frame(n, x, y, s, ball=None):
    if ball is None:
        return f'<Frame N="{n}" T="t" X="{x:.2f}" Y="{y:.2f}" D="0.00" S="{s:.2f}" A="0.00" M="1"/>'
    possession, status = ball
    return (
        f'<Frame N="{n}" T="t" X="{x:.2f}" Y="{y:.2f}" Z="0.10" D="0.00" S="{s:.2f}" A="0.00" M="1" '
        f'BallPossession="{possession}" BallStatus="{status}"/>'
    )


def _positions():
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest><Positions>"]
    for section, first in (("firstHalf", 10000), ("secondHalf", 100000)):
        lines.append(f'<FrameSet GameSection="{section}" MatchId="M" TeamId="CLU-H" PersonId="P1">')
        lines += [_frame(first + i, 1.0, 2.0, 3.0) for i in range(10)]
        lines.append("</FrameSet>")
        lines.append(f'<FrameSet GameSection="{section}" MatchId="M" TeamId="CLU-A" PersonId="P2">')
        lines += [_frame(first + i, -1.0, -2.0, 1.0) for i in range(5, 10)]
        lines.append("</FrameSet>")
        lines.append(f'<FrameSet GameSection="{section}" MatchId="M" TeamId="REF" PersonId="R1">')
        lines += [_frame(first + i, 0.0, 0.0, 0.0) for i in range(10)]
        lines.append("</FrameSet>")
        lines.append(f'<FrameSet GameSection="{section}" MatchId="M" TeamId="BALL" PersonId="B">')
        codes = [(1, 1)] * 4 + [(2, 1)] * 3 + [(2, 0)] * 3
        lines += [_frame(first + i, 0.0, 0.0, 5.0, codes[i]) for i in range(10)]
        lines.append("</FrameSet>")
    lines.append("</Positions></PutDataRequest>")
    return "\n".join(lines) + "\n"


@pytest.fixture
def match(tmp_path):
    info = tmp_path / "info.xml"
    positions = tmp_path / "positions.xml"
    info.write_text(INFO, encoding="utf-8")
    positions.write_text(_positions(), encoding="utf-8")
    return read_match(info, positions)


def test_halves_are_separated_and_referee_dropped(match):
    assert match.n_frames == 21
    assert match.half.tolist() == [1] * 10 + [0] + [2] * 10
    assert match.slot_team.tolist() == [0, 1]
    assert match.pitch == (100.0, 60.0)


def test_coordinates_move_to_corner_origin(match):
    assert match.players[0, 0, 0] == pytest.approx(51.0)
    assert match.players[0, 0, 1] == pytest.approx(28.0)
    assert np.isnan(match.players[0, 1, 0])
    assert match.players[5, 1, 0] == pytest.approx(49.0)
    assert match.ball[0].tolist() == pytest.approx([50.0, 30.0])


def test_native_labels(match):
    labels = native_labels(match)
    half = [H] * 4 + [A] * 3 + [D] * 3
    assert labels.tolist() == half + [X] + half


def test_event_labels_switch_only_on_control(match):
    events = pd.DataFrame(
        {
            "frame_id": [10000, 10002, 10003, 10005, 100001, 100002],
            "period_id": [1, 1, 1, 1, 2, 2],
            "player_id": ["home_1", "away_2", "away_2", "away_2", "away_2", "home_1"],
            "spadl_type": ["pass", "bad_touch", "clearance", "control", "pass", "tackle"],
            "success": [True, True, True, True, True, True],
        }
    )
    labels, report = event_labels(match, events)
    assert labels[:10].tolist() == [H] * 5 + [A] * 2 + [D] * 3
    assert labels[10] == X
    assert labels[11:].tolist() == [A] * 7 + [D] * 3
    assert report.n_control == 3
    assert report.unknown_types == ()


def test_official_metric_on_identical_labels(match):
    labels = native_labels(match)
    result = evaluate(labels, labels)
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_parser_is_order_independent_and_converts_speed(tmp_path):
    info = tmp_path / "info.xml"
    positions = tmp_path / "positions.xml"
    info.write_text(INFO, encoding="utf-8")
    lines = ['<?xml version="1.0"?>', "<PutDataRequest><Positions>"]
    for section, first in (("firstHalf", 10000), ("secondHalf", 100000)):
        lines.append(f'<FrameSet GameSection="{section}" MatchId="M" TeamId="CLU-H" PersonId="P1">')
        lines.append(f'<Frame S="36.00" Y="2.00" X="1.00" N="{first}" T="t"/>')
        lines.append("</FrameSet>")
        lines.append(f'<FrameSet GameSection="{section}" MatchId="M" TeamId="BALL" PersonId="B">')
        lines.append(f'<Frame BallStatus="1" BallPossession="2" N="{first}" X="0" Y="0" Z="0.1" S="0"/>')
        lines.append("</FrameSet>")
    lines.append("</Positions></PutDataRequest>")
    positions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = read_match(info, positions)
    assert parsed.players[0, 0, 2] == pytest.approx(10.0)
    assert parsed.ball_possession[0] == 2
    assert parsed.ball_status[0] == 1


def test_parser_rejects_unreadable_frame(tmp_path):
    info = tmp_path / "info.xml"
    positions = tmp_path / "positions.xml"
    info.write_text(INFO, encoding="utf-8")
    positions.write_text(
        '<FrameSet GameSection="firstHalf" MatchId="M" TeamId="CLU-H" PersonId="P1">\n<Frame N="1" Y="2"/>\n</FrameSet>\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_match(info, positions)


def test_restart_event_just_before_alive_run_sets_team(match):
    events = pd.DataFrame(
        {
            "frame_id": [100000 - 0, 100001],
            "period_id": [2, 2],
            "player_id": ["home_1", "away_2"],
            "spadl_type": ["throw_in", "control"],
            "success": [True, True],
        }
    )
    status = match.ball_status.copy()
    status[11] = 0
    shifted = type(match)(**{**match.__dict__, "ball_status": status})
    labels, _ = event_labels(shifted, events)
    assert labels[11] == D
    assert labels[12] == A
    events.loc[1, "frame_id"] = 100003
    labels, _ = event_labels(shifted, events)
    assert labels[12:14].tolist() == [H, H]
    assert labels[14] == A


def test_native_labels_carry_last_team_over_unknown_possession(match):
    codes = match.ball_possession.copy()
    codes[2] = 0
    patched = type(match)(**{**match.__dict__, "ball_possession": codes})
    labels = native_labels(patched)
    assert labels[2] == H
    assert X not in labels[:10].tolist()
