"""End to end behavior tests.

Train the actionness network on synthetic sequences, run the proposal pipeline
on held out sequences, then check that the planted boundaries are recovered
within a frame tolerance and that the temporal IoU of proposals against ground
truth beats a chance baseline.
"""

import numpy as np
import pytest
import torch

from src.data import make_dataset, make_sequence
from src.proposals import proposals_from_actionness, match_proposals, temporal_iou
from src.train import train_model

IN_DIM = 16


@pytest.fixture(scope="module")
def trained():
    train_ds = make_dataset(n=64, dim=IN_DIM, seed=0)
    model, history = train_model(
        dataset=train_ds, in_dim=IN_DIM, epochs=60, lr=1e-2, seed=0
    )
    # Loss should have decreased over training.
    assert history["loss"][-1] < history["loss"][0]
    return model


def _eval_sequence(model, seed):
    feats, action, segs = make_sequence(dim=IN_DIM, seed=seed)
    x = torch.from_numpy(feats).unsqueeze(0)  # (1, T, D)
    probs = model.predict_actionness(x).squeeze(0).numpy()
    props = proposals_from_actionness(probs, threshold=0.5, min_len=3)
    return action, segs, props, probs


def test_actionness_separates_frames(trained):
    # Mean predicted actionness inside true segments exceeds outside.
    inside, outside = [], []
    for seed in range(200, 220):
        action, segs, props, probs = _eval_sequence(trained, seed)
        mask = action > 0.5
        if mask.any() and (~mask).any():
            inside.append(probs[mask].mean())
            outside.append(probs[~mask].mean())
    assert np.mean(inside) > np.mean(outside) + 0.2


def test_boundaries_within_tolerance(trained):
    tol = 4  # frames
    hits, total = 0, 0
    for seed in range(300, 340):
        action, segs, props, probs = _eval_sequence(trained, seed)
        for seg in segs:
            total += 1
            # Does some proposal boundary land near this segment's edges?
            for p in props:
                if abs(p.start - seg.start) <= tol and abs(p.end - seg.end) <= tol:
                    hits += 1
                    break
    assert total > 0
    recall = hits / total
    assert recall >= 0.8, f"boundary recall {recall:.2f} below 0.8"


def test_iou_beats_chance(trained):
    # Collect mean best IoU of model proposals against ground truth.
    model_ious = []
    seg_lengths = []
    seq_len = None
    for seed in range(400, 440):
        action, segs, props, probs = _eval_sequence(trained, seed)
        if seq_len is None:
            seq_len = len(action)
        _, best = match_proposals(props, segs, iou_thresh=0.5)
        model_ious.extend(best)
        seg_lengths.extend(s.length for s in segs)

    mean_model_iou = float(np.mean(model_ious))

    # Chance baseline: random proposals of the same typical length placed
    # uniformly at random, scored by best IoU against the same ground truth.
    rng = np.random.default_rng(123)
    chance_ious = []
    avg_len = int(np.mean(seg_lengths))
    for seed in range(400, 440):
        action, segs, _, _ = _eval_sequence(trained, seed)
        T = len(action)
        for seg in segs:
            best = 0.0
            for _ in range(5):  # a few random guesses, take the best
                start = int(rng.integers(0, max(1, T - avg_len)))
                end = min(T, start + avg_len)
                best = max(best, temporal_iou((start, end), (seg.start, seg.end)))
            chance_ious.append(best)
    mean_chance_iou = float(np.mean(chance_ious))

    assert mean_model_iou > 0.5, f"model mean IoU {mean_model_iou:.3f} too low"
    assert mean_model_iou > 2.0 * mean_chance_iou, (
        f"model IoU {mean_model_iou:.3f} does not clearly beat chance "
        f"{mean_chance_iou:.3f}"
    )


def test_detection_recall_at_iou_half(trained):
    matched_total, gt_total = 0, 0
    for seed in range(500, 540):
        action, segs, props, probs = _eval_sequence(trained, seed)
        matched, _ = match_proposals(props, segs, iou_thresh=0.5)
        matched_total += matched
        gt_total += len(segs)
    recall = matched_total / gt_total
    assert recall >= 0.8, f"detection recall@0.5 {recall:.2f} below 0.8"
