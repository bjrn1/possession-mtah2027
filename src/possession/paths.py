import os
from dataclasses import dataclass
from pathlib import Path

GAME = "BAR-ATH"
FPS = 25
PITCH_METRES = (105.3, 68.0)
N_FRAMES = {"tactical": 199319}
HALF_VIDEO_BLOCKS = ((1, 11185, 90651), (2, 110808, 190197))

SUFFIXES = {
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
}

ASSETS = {
    "tactical": ("atd_meta", "atd_raw"),
}


@dataclass(frozen=True)
class Bundle:
    root: Path
    source: str

    def path(self, asset: str) -> Path:
        if asset not in ASSETS[self.source]:
            raise KeyError(f"source {self.source!r} has no asset {asset!r}")
        return self.root / self.source / f"{GAME}_{self.source}{SUFFIXES[asset]}"

    def missing(self) -> list[Path]:
        return [p for p in (self.path(a) for a in ASSETS[self.source]) if not p.is_file()]

    @property
    def n_frames(self) -> int:
        return N_FRAMES[self.source]


def data_root(explicit: str | os.PathLike | None = None) -> Path:
    value = explicit or os.environ.get("MTAH_DATA_DIR")
    if not value:
        raise RuntimeError("set MTAH_DATA_DIR or pass --data-dir")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def bundle(explicit: str | os.PathLike | None = None, source: str = "tactical") -> Bundle:
    if source not in ASSETS:
        raise KeyError(f"unknown source {source!r}")
    return Bundle(data_root(explicit), source)
