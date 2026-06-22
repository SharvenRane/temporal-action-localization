# temporal-action-localization

Find where actions happen inside a long, untrimmed sequence. The model reads a
stream of frame features, predicts a per frame actionness score (how likely the
frame sits inside some action), and a proposal stage turns that score into
segment boundaries with start and end times.

Everything runs on CPU with synthetic data, so the whole thing trains and tests
in a couple of seconds with no downloads and no API keys. The synthetic
generator plants action segments into a noisy background, which gives exact
ground truth boundaries to measure against.

## The idea

Untrimmed video localization usually has two parts. First a frame level signal
tells you which moments look like activity. Then a proposal stage groups the
active moments into candidate segments and ranks them. This repo keeps that
shape but swaps real video features for a tractable synthetic stand in so the
behavior is easy to verify.

1. **Data.** Each sequence is a `(T, D)` feature matrix. Background frames are
   low energy Gaussian noise. Inside a planted segment the frames carry a class
   specific direction in feature space with a higher amplitude, shaped by a
   smooth envelope so the interior is clearly action while the edges fade. The
   generator returns the features, a per frame actionness target, and the list
   of planted segments. Segments never overlap and always have a background gap
   between them, so the boundaries are well defined.

2. **Model.** A small 1D temporal convolutional network maps `(B, T, D)` to a
   per frame actionness logit `(B, T)`. Padding keeps the temporal length fixed
   end to end, so prediction frame `t` lines up exactly with input frame `t`.
   The receptive field spans several neighbouring frames, which lets the model
   use local context to place boundaries. This is a real trainable network, not
   a stub.

3. **Proposals.** The actionness curve is smoothed, thresholded, and split into
   contiguous runs above the threshold. Each run becomes a candidate segment
   scored by its mean actionness. Greedy temporal non maximum suppression drops
   overlapping duplicates. The result is a ranked list of `[start, end)`
   proposals.

4. **Scoring.** Temporal IoU compares a proposal interval to a ground truth
   interval. Greedy matching assigns each ground truth segment to at most one
   proposal above an IoU threshold, which gives detection recall, and records
   the best IoU per segment, which measures localization quality.

## Layout

```
src/
  data.py        synthetic sequence and dataset generation
  model.py       ActionnessNet, the temporal conv net
  proposals.py   actionness to proposals, temporal IoU, NMS, matching
  train.py       training loop with class balanced BCE loss
tests/
  test_data.py          generator shapes, determinism, energy gap
  test_model.py         forward shapes, probability range, gradients
  test_proposals.py     IoU values, run extraction, NMS, matching
  test_localization.py  end to end: boundaries recovered, IoU beats chance
```

## Install and run

```
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```

## What the tests check

The unit tests confirm the generator, model, and proposal pipeline behave as
specified: shapes and dtypes, that action frames carry more energy than
background, that temporal IoU returns the right values on known intervals, that
NMS suppresses overlaps, and that greedy matching never double counts.

The end to end tests train the network on 64 synthetic sequences and then
evaluate on held out sequences with unseen seeds. They require that predicted
actionness is clearly higher inside true segments than outside, that proposal
boundaries land within four frames of the planted boundaries for most segments,
that detection recall at IoU 0.5 stays high, and that the mean temporal IoU of
the proposals beats a random placement baseline by a wide margin.

## Results from the included run

These are figures produced by training with the default settings on this
machine, seed 0, 60 epochs on CPU. They are reproduced by the test suite.

- Training loss fell from about 0.77 to about 0.001.
- Mean best temporal IoU of proposals against ground truth: about 0.89.
- Detection recall at IoU 0.5: about 0.99.

Your exact numbers can move slightly with the BLAS and PyTorch build, but the
behavior the tests assert holds with comfortable margin.

## Notes

The synthetic features stand in for a heavy pretrained video backbone purely so
the project runs offline in seconds. The actionness network, the proposal
extraction, the temporal IoU and NMS, and the matching are all real and would
sit unchanged on top of features from a true backbone.
