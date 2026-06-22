import numpy as np

from src.data import make_sequence, make_dataset, Segment


def test_shapes_and_dtypes():
    feats, action, segs = make_sequence(length=100, dim=16, seed=1)
    assert feats.shape == (100, 16)
    assert action.shape == (100,)
    assert feats.dtype == np.float32
    assert action.dtype == np.float32
    assert all(isinstance(s, Segment) for s in segs)


def test_actionness_matches_segments():
    feats, action, segs = make_sequence(length=120, dim=16, seed=7)
    expected = np.zeros(120, dtype=np.float32)
    for s in segs:
        expected[s.start : s.end] = 1.0
    assert np.array_equal(action, expected)
    # Every active frame belongs to exactly one segment.
    assert action.sum() == sum(s.length for s in segs)


def test_segments_non_overlapping_with_gap():
    _, _, segs = make_sequence(length=200, dim=8, max_segments=4, gap=4, seed=3)
    segs = sorted(segs, key=lambda s: s.start)
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start
        # At least one background frame separates them.
        assert b.start - a.end >= 1


def test_action_frames_have_more_energy():
    # Interior action frames should carry more signal energy than background.
    feats, action, segs = make_sequence(
        length=160, dim=16, signal=3.0, noise=1.0, seed=11
    )
    energy = (feats ** 2).sum(axis=1)
    mask = action > 0.5
    assert mask.any()
    assert (~mask).any()
    assert energy[mask].mean() > energy[~mask].mean()


def test_determinism():
    a = make_sequence(seed=42)
    b = make_sequence(seed=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_dataset_distinct_sequences():
    ds = make_dataset(n=5, seed=0)
    assert len(ds) == 5
    # Distinct seeds give distinct feature matrices.
    assert not np.array_equal(ds[0][0], ds[1][0])
