"""Turn a per frame actionness signal into action segment proposals.

The pipeline thresholds the actionness curve, groups contiguous runs above the
threshold into candidate segments, scores each candidate by its mean
actionness, then applies temporal non maximum suppression. Helpers compute
temporal IoU and match proposals to ground truth segments.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .data import Segment


@dataclass(frozen=True)
class Proposal:
    """A predicted action segment with a confidence score."""

    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average smoothing with reflection at the edges."""
    if win <= 1:
        return x.astype(np.float64)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(x.astype(np.float64), pad, mode="reflect")
    kernel = np.ones(win) / win
    return np.convolve(padded, kernel, mode="valid")


def proposals_from_actionness(
    actionness: Sequence[float],
    threshold: float = 0.5,
    min_len: int = 3,
    smooth_win: int = 3,
    nms_iou: float = 0.5,
) -> List[Proposal]:
    """Extract scored proposals from a 1D actionness signal.

    Parameters
    ----------
    actionness : per frame scores, typically probabilities in [0, 1]
    threshold : frames at or above this are considered active
    min_len : drop runs shorter than this many frames
    smooth_win : moving average window applied before thresholding
    nms_iou : temporal IoU above which overlapping proposals are suppressed

    Returns proposals sorted by descending score, each as a half open interval
    [start, end).
    """
    a = np.asarray(actionness, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("actionness must be 1D")
    if a.size == 0:
        return []

    sm = _smooth(a, smooth_win)
    active = sm >= threshold

    proposals: List[Proposal] = []
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            start, end = i, j  # half open [start, end)
            if end - start >= min_len:
                score = float(a[start:end].mean())
                proposals.append(Proposal(start=start, end=end, score=score))
            i = j
        else:
            i += 1

    return nms_1d(proposals, iou_thresh=nms_iou)


def temporal_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Temporal intersection over union of two half open intervals.

    Each interval is (start, end) covering [start, end). Returns 0.0 when the
    union is empty or the intervals do not overlap.
    """
    a_start, a_end = a
    b_start, b_end = b
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_1d(proposals: List[Proposal], iou_thresh: float = 0.5) -> List[Proposal]:
    """Greedy temporal non maximum suppression.

    Keeps the highest scoring proposal, removes any remaining proposal whose
    temporal IoU with a kept one exceeds the threshold, repeats. Returns the
    kept proposals sorted by descending score.
    """
    order = sorted(proposals, key=lambda p: p.score, reverse=True)
    kept: List[Proposal] = []
    for p in order:
        if all(
            temporal_iou((p.start, p.end), (k.start, k.end)) <= iou_thresh
            for k in kept
        ):
            kept.append(p)
    return kept


def match_proposals(
    proposals: Sequence[Proposal],
    segments: Sequence[Segment],
    iou_thresh: float = 0.5,
) -> Tuple[int, List[float]]:
    """Greedily match proposals to ground truth segments by temporal IoU.

    Each ground truth segment is matched to at most one proposal, choosing the
    available proposal with the largest IoU above the threshold. Returns the
    number of matched ground truth segments and, for every ground truth
    segment, the best IoU achieved by any proposal (regardless of threshold),
    which is useful for measuring localization quality and recall.
    """
    best_ious: List[float] = []
    used = [False] * len(proposals)
    matched = 0

    for seg in segments:
        best_iou = 0.0
        best_idx = -1
        for idx, p in enumerate(proposals):
            iou = temporal_iou((p.start, p.end), (seg.start, seg.end))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        best_ious.append(best_iou)
        if best_idx >= 0 and best_iou >= iou_thresh and not used[best_idx]:
            used[best_idx] = True
            matched += 1

    return matched, best_ious


def boundaries(proposals: Sequence[Proposal]) -> List[Tuple[int, int]]:
    """Convenience: list of (start, end) boundary pairs from proposals."""
    return [(p.start, p.end) for p in proposals]
