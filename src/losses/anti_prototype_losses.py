# coding: utf-8
import torch
import torch.nn.functional as F


def anti_separation_loss(z_neg, z_pos, margin):
    cosine = F.cosine_similarity(z_neg, z_pos, dim=-1, eps=1e-8)
    return F.relu(cosine - margin).mean()


def anti_norm_loss(z_neg, x0):
    return torch.sum((z_neg - x0) ** 2, dim=-1).mean()


def anti_calibration_loss(z_neg, target):
    cosine = F.cosine_similarity(z_neg, target, dim=-1, eps=1e-8)
    return 1.0 - cosine.mean()


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 4
    embed_dim = 12
    z_neg = torch.randn(batch_size, embed_dim)
    z_pos = torch.randn(batch_size, embed_dim)
    x0 = torch.randn(batch_size, embed_dim)
    target = torch.randn(batch_size, embed_dim)

    sep = anti_separation_loss(z_neg, z_pos, margin=0.2)
    norm = anti_norm_loss(z_neg, x0)
    calib = anti_calibration_loss(z_neg, target)

    assert sep.dim() == 0
    assert norm.dim() == 0
    assert calib.dim() == 0
    print('anti_prototype_losses smoke test passed.')
