import numpy as np
import pytest

from possession.io_atd import AWAY, HOME, read_meta, read_tracking, scan_raw

META = """<?xml version="1.0"?>
<main><Metadata><Teams>
<Team id="T1"><Name>Home</Name></Team>
<Team id="T2"><Name>Away</Name></Team>
</Teams><Players>
<Player id="P1" teamId="T1"/>
<Player id="P2" teamId="T2"/>
<Player id="P3" teamId="T1"/>
</Players></Metadata></main>
"""

RAW = (
    "1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,0.0:0.5,0.5,0.0\n"
    "2:0.6,0.5,2.0;0.4,0.4,1.5;NaN,NaN,NaN:NaN,NaN,NaN\n"
    "3:NaN,NaN,NaN;0.4,0.5,1.0;0.2,0.2,0.5:0.25,0.75,3.0\n"
)


@pytest.fixture
def files(tmp_path):
    meta = tmp_path / "meta.xml"
    raw = tmp_path / "raw.txt"
    meta.write_text(META, encoding="utf-8")
    raw.write_text(RAW, encoding="utf-8")
    return meta, raw


def test_meta_maps_first_team_to_home(files):
    meta = read_meta(files[0])
    assert meta.slot_team.tolist() == [HOME, AWAY, HOME]


def test_tracking_is_indexed_by_video_frame_in_metres(files):
    tracking = read_tracking(files[0], files[1], n_frames=3, pitch=(100.0, 50.0))
    assert tracking.present.all()
    assert tracking.players[0, 0, 0] == pytest.approx(50.0)
    assert tracking.players[0, 0, 1] == pytest.approx(25.0)
    assert tracking.players[0, 0, 2] == pytest.approx(1.0)
    assert np.isnan(tracking.players[0, 1, 0])
    assert tracking.ball_seen.tolist() == [True, False, True]
    assert tracking.ball[2].tolist() == pytest.approx([25.0, 37.5])


def test_tracking_rejects_frame_outside_range(files):
    with pytest.raises(ValueError):
        read_tracking(files[0], files[1], n_frames=2, pitch=(100.0, 50.0))


def test_scan_reports_structure(files):
    scan = scan_raw(files[1])
    assert scan["records"] == 3
    assert scan["contiguous"]
    assert scan["slots_per_line"] == [3]
    assert scan["ball_fields"] == [3]
