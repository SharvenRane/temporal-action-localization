"""Per frame actionness model.

A small temporal convolutional network reads a sequence of frame features and
predicts a per frame actionness logit. The receptive field spans several
neighbouring frames so the model can use local context to decide whether a
frame sits inside an action and where boundaries lie. This is a real, trainable
network, not a stand in.
"""

import torch
import torch.nn as nn


class ActionnessNet(nn.Module):
    """1D temporal conv net mapping (B, T, D) features to (B, T) actionness logits.

    Padding is set so the temporal length is preserved end to end, which keeps
    the per frame alignment between input frames and output predictions exact.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 32,
        num_layers: int = 3,
        kernel_size: int = 5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")
        self.in_dim = in_dim
        pad = kernel_size // 2

        layers = []
        ch = in_dim
        for _ in range(num_layers):
            layers.append(nn.Conv1d(ch, hidden, kernel_size, padding=pad))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            ch = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> logits: (B, T)."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got shape {tuple(x.shape)}")
        # Conv1d wants (B, C, T).
        h = x.transpose(1, 2)
        h = self.backbone(h)
        logits = self.head(h)  # (B, 1, T)
        return logits.squeeze(1)  # (B, T)

    @torch.no_grad()
    def predict_actionness(self, x: torch.Tensor) -> torch.Tensor:
        """Return per frame actionness probabilities in [0, 1]."""
        self.eval()
        return torch.sigmoid(self.forward(x))
