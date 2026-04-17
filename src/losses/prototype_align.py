# coding: utf-8
import torch
import torch.nn.functional as F


def prototype_cosine_alignment_loss(z_proto, target_proto, eps=1e-8):
    """Compute cosine alignment loss between generated and target prototypes."""
    cosine = F.cosine_similarity(z_proto, target_proto, dim=-1, eps=eps)
    return 1.0 - cosine.mean()


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 4
    embed_dim = 12

    z_proto = torch.randn(batch_size, embed_dim)
    target_proto = torch.randn(batch_size, embed_dim)
    loss = prototype_cosine_alignment_loss(z_proto, target_proto)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    print('prototype_cosine_alignment_loss smoke test passed.')
