"""Train the actionness network on synthetic sequences.

Sequences in this generator share a fixed length, so a batch is a stack of
(T, D) feature matrices with matching (T,) actionness targets. Training is a
straightforward per frame binary classification with BCE loss.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import Segment, make_dataset
from .model import ActionnessNet


def _to_batch(
    dataset: List[Tuple[np.ndarray, np.ndarray, List[Segment]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = np.stack([d[0] for d in dataset], axis=0)  # (N, T, D)
    targs = np.stack([d[1] for d in dataset], axis=0)  # (N, T)
    return (
        torch.from_numpy(feats).float(),
        torch.from_numpy(targs).float(),
    )


def train_model(
    dataset: Optional[List[Tuple[np.ndarray, np.ndarray, List[Segment]]]] = None,
    in_dim: int = 16,
    epochs: int = 60,
    lr: float = 1e-2,
    batch_size: int = 16,
    hidden: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[ActionnessNet, dict]:
    """Train an :class:`ActionnessNet` and return it with a small history dict.

    If ``dataset`` is None a default synthetic training set is generated. The
    feature dimension ``in_dim`` must match the data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset is None:
        dataset = make_dataset(n=64, dim=in_dim, seed=seed)

    X, Y = _to_batch(dataset)
    n = X.shape[0]

    model = ActionnessNet(in_dim=in_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Balance the loss because background frames outnumber action frames.
    pos = float(Y.sum().item())
    neg = float(Y.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"loss": []}
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = X[idx]
            yb = Y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1
        history["loss"].append(epoch_loss / max(nb, 1))
        if verbose:
            print(f"loss {history['loss'][-1]:.4f}")

    return model, history
