# coding: utf-8
import torch


def flow_matching_mse_loss(v_pred, v_target):
    """Compute mean squared error for flow matching velocity targets."""
    return torch.mean((v_pred - v_target) ** 2)


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 4
    embed_dim = 12

    velocity_pred = torch.randn(batch_size, embed_dim)
    velocity_target = torch.randn(batch_size, embed_dim)
    loss = flow_matching_mse_loss(velocity_pred, velocity_target)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    print('flow_matching_mse_loss smoke test passed.')
