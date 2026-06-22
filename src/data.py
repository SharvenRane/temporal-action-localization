"""Synthetic untrimmed sequence generator with planted action segments.

Each sequence is a feature matrix of shape (T, D). Background frames are low
energy noise. Inside a planted action segment the frames carry a class specific
pattern with higher energy, so a model can learn per frame actionness (is this
frame inside any action) plus where the boundaries fall.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Segment:
    """A planted action segment.

    start and end are frame indices. The segment covers frames in the half open
    interval [start, end), so its length in frames is end - start.
    """

    start: int
    end: int
    label: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _class_pattern(label: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """A fixed direction in feature space for an action class.

    The direction is deterministic given the label so every instance of the
    same class looks alike up to noise. A separate rng draws the noise.
    """
    seed_rng = np.random.default_rng(1000 + label)
    direction = seed_rng.standard_normal(dim)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


def make_sequence(
    length: int = 120,
    dim: int = 16,
    num_classes: int = 3,
    max_segments: int = 3,
    min_seg_len: int = 8,
    max_seg_len: int = 24,
    gap: int = 4,
    signal: float = 3.0,
    noise: float = 1.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[Segment]]:
    """Generate one untrimmed sequence with non overlapping planted segments.

    Returns
    -------
    features : float32 array, shape (length, dim)
    actionness : float32 array, shape (length,), 1.0 inside any segment else 0.0
    segments : list of Segment, sorted by start, non overlapping with a gap

    The placement guarantees at least ``gap`` background frames between
    consecutive segments and before the first and after the last segment, so
    boundaries are well defined and recoverable.
    """
    if min_seg_len < 1:
        raise ValueError("min_seg_len must be positive")
    if max_seg_len < min_seg_len:
        raise ValueError("max_seg_len must be >= min_seg_len")

    rng = np.random.default_rng(seed)

    features = noise * rng.standard_normal((length, dim)).astype(np.float64)
    actionness = np.zeros(length, dtype=np.float64)
    segments: List[Segment] = []

    cursor = gap
    n_target = int(rng.integers(1, max_segments + 1))

    for _ in range(n_target):
        seg_len = int(rng.integers(min_seg_len, max_seg_len + 1))
        # Need room for this segment plus a trailing gap.
        if cursor + seg_len + gap > length:
            break
        # Optionally slide the start forward by a random slack so segments are
        # not all flush against each other.
        max_slack = length - (cursor + seg_len + gap)
        slack = int(rng.integers(0, max(1, min(max_slack, gap) + 1)))
        start = cursor + slack
        end = start + seg_len
        label = int(rng.integers(0, num_classes))

        direction = _class_pattern(label, dim, rng)
        # Inject the class pattern with a smooth amplitude envelope so the
        # interior is clearly action while edges fade. The envelope never drops
        # to zero inside the segment, keeping every interior frame above
        # background energy.
        t = np.linspace(0.0, np.pi, seg_len)
        envelope = 0.5 + 0.5 * np.sin(t) ** 0.5  # in [0.5, 1.0]
        amp = signal * envelope
        pattern = amp[:, None] * direction[None, :]
        features[start:end] += pattern
        actionness[start:end] = 1.0

        segments.append(Segment(start=start, end=end, label=label))
        cursor = end + gap

    return (
        features.astype(np.float32),
        actionness.astype(np.float32),
        segments,
    )


def make_dataset(
    n: int = 64,
    seed: int = 0,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray, List[Segment]]]:
    """Generate a list of ``n`` sequences with distinct seeds."""
    out = []
    for i in range(n):
        out.append(make_sequence(seed=seed + i, **kwargs))
    return out
