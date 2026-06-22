import torch

from src.model import ActionnessNet


def test_forward_preserves_length():
    net = ActionnessNet(in_dim=16, hidden=8)
    x = torch.randn(4, 50, 16)
    out = net(x)
    assert out.shape == (4, 50)


def test_predict_actionness_in_unit_range():
    net = ActionnessNet(in_dim=16, hidden=8)
    x = torch.randn(2, 30, 16)
    p = net.predict_actionness(x)
    assert p.shape == (2, 30)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)


def test_variable_length_inputs():
    net = ActionnessNet(in_dim=16, hidden=8)
    for T in (20, 75, 128):
        out = net(torch.randn(1, T, 16))
        assert out.shape == (1, T)


def test_gradients_flow():
    net = ActionnessNet(in_dim=16, hidden=8)
    x = torch.randn(3, 40, 16)
    y = torch.randint(0, 2, (3, 40)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)
