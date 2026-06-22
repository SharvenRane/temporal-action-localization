import numpy as np

from src.data import Segment
from src.proposals import (
    Proposal,
    proposals_from_actionness,
    temporal_iou,
    match_proposals,
    nms_1d,
)


def test_temporal_iou_known_values():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (10, 20)) == 0.0  # touching, no overlap
    assert temporal_iou((0, 10), (20, 30)) == 0.0  # disjoint
    # overlap 5, union 15
    assert abs(temporal_iou((0, 10), (5, 15)) - (5 / 15)) < 1e-9


def test_proposals_recover_clean_segments():
    a = np.zeros(100, dtype=np.float32)
    a[10:30] = 1.0
    a[60:80] = 1.0
    props = proposals_from_actionness(a, threshold=0.5, min_len=3)
    bounds = sorted((p.start, p.end) for p in props)
    assert bounds == [(10, 30), (60, 80)]


def test_short_runs_filtered():
    a = np.zeros(50, dtype=np.float32)
    a[5:7] = 1.0  # length 2, below min_len
    a[20:35] = 1.0  # length 15, kept
    props = proposals_from_actionness(a, threshold=0.5, min_len=3)
    bounds = sorted((p.start, p.end) for p in props)
    assert bounds == [(20, 35)]


def test_nms_suppresses_overlap():
    p1 = Proposal(0, 10, score=0.9)
    p2 = Proposal(1, 11, score=0.5)  # high IoU with p1, lower score
    p3 = Proposal(50, 60, score=0.7)
    kept = nms_1d([p1, p2, p3], iou_thresh=0.5)
    kept_bounds = {(p.start, p.end) for p in kept}
    assert (0, 10) in kept_bounds
    assert (50, 60) in kept_bounds
    assert (1, 11) not in kept_bounds


def test_match_proposals_counts_and_iou():
    segs = [Segment(10, 30, 0), Segment(60, 80, 1)]
    props = [Proposal(11, 29, 0.9), Proposal(60, 80, 0.8)]
    matched, ious = match_proposals(props, segs, iou_thresh=0.5)
    assert matched == 2
    assert len(ious) == 2
    assert all(i >= 0.5 for i in ious)
    # Perfect proposal gives IoU 1.0.
    assert abs(ious[1] - 1.0) < 1e-9


def test_match_no_double_count():
    # Two ground truths but only one proposal that can match one of them.
    segs = [Segment(10, 30, 0), Segment(11, 31, 0)]
    props = [Proposal(10, 30, 0.9)]
    matched, _ = match_proposals(props, segs, iou_thresh=0.5)
    assert matched == 1
