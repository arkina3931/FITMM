import torch
import torch.nn as nn


class SoftPopularNicheSplitter(nn.Module):
    def __init__(self, num_bands, embed_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        condition_dim = embed_dim + num_bands + 1
        self.pop_router = self._build_router(condition_dim)
        self.niche_router = self._build_router(condition_dim)

    def _build_router(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_bands),
        )

    @staticmethod
    def _stable_softmax(logits):
        shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
        return torch.softmax(shifted_logits, dim=-1)

    def forward(self, band_components, task_emb, prior=None):
        if len(band_components) != self.num_bands:
            raise ValueError(f'Expected {self.num_bands} bands, but received {len(band_components)}.')

        band_tensor = torch.stack(band_components, dim=1)
        band_norms = torch.norm(band_tensor, p=2, dim=-1)

        if prior is None:
            prior = task_emb.new_zeros(task_emb.size(0), 1)
        elif prior.dim() == 1:
            prior = prior.unsqueeze(-1)

        if prior.size(0) != task_emb.size(0):
            raise ValueError('Prior batch size must match task embedding batch size.')

        router_condition = torch.cat([task_emb, band_norms, prior], dim=-1)

        pop_logits = self.pop_router(router_condition)
        niche_logits = self.niche_router(router_condition)

        pop_weights = self._stable_softmax(pop_logits)
        niche_weights = self._stable_softmax(niche_logits)

        pop_repr = torch.sum(pop_weights.unsqueeze(-1) * band_tensor, dim=1)
        niche_repr = torch.sum(niche_weights.unsqueeze(-1) * band_tensor, dim=1)

        routing_info = {
            'pop_weights': pop_weights,
            'niche_weights': niche_weights,
            'band_norms': band_norms,
        }
        return pop_repr, niche_repr, routing_info


if __name__ == '__main__':
    torch.manual_seed(2026)

    num_nodes = 5
    num_bands = 3
    embed_dim = 12

    bands = [torch.randn(num_nodes, embed_dim) for _ in range(num_bands)]
    task_emb = torch.randn(num_nodes, embed_dim)
    prior = torch.rand(num_nodes, 1)

    splitter = SoftPopularNicheSplitter(
        num_bands=num_bands,
        embed_dim=embed_dim,
        hidden_dim=16,
        dropout=0.1,
    )
    pop_repr, niche_repr, routing_info = splitter(bands, task_emb, prior)

    assert pop_repr.shape == (num_nodes, embed_dim)
    assert niche_repr.shape == (num_nodes, embed_dim)
    assert routing_info['pop_weights'].shape == (num_nodes, num_bands)
    assert routing_info['niche_weights'].shape == (num_nodes, num_bands)
    assert torch.allclose(
        routing_info['pop_weights'].sum(dim=-1),
        torch.ones(num_nodes),
        atol=1e-6,
    )
    assert torch.allclose(
        routing_info['niche_weights'].sum(dim=-1),
        torch.ones(num_nodes),
        atol=1e-6,
    )
    print('SoftPopularNicheSplitter smoke test passed.')
