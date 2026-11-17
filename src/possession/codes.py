import numpy as np

H = 0
A = 1
D = 2
X = 3
LETTERS = np.array(["H", "A", "D", "X"])


def to_letters(codes: np.ndarray) -> np.ndarray:
    return LETTERS[codes]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
