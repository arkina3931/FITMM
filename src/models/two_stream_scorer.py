import torch
import torch.nn as nn


class TwoStreamScorer(nn.Module):
    @staticmethod
    def _require(value, name):
        if value is None:
            raise ValueError(f'Missing required tensor: {name}')
        return value

    def score_full(
        self,
        mode='residual',
        user_out=None,
        item_out=None,
        user_pop=None,
        item_pop=None,
        user_niche=None,
        item_niche=None,
        alpha=1.0,
        beta=1.0,
    ):
        if mode == 'residual':
            user_out = self._require(user_out, 'user_out')
            item_out = self._require(item_out, 'item_out')
            return user_out @ item_out.t()

        if mode == 'two_stream':
            user_pop = self._require(user_pop, 'user_pop')
            item_pop = self._require(item_pop, 'item_pop')
            user_niche = self._require(user_niche, 'user_niche')
            item_niche = self._require(item_niche, 'item_niche')
            return alpha * (user_pop @ item_pop.t()) + beta * (user_niche @ item_niche.t())

        raise ValueError(f'Unknown score mode: {mode}')

    def score_pair(
        self,
        mode='residual',
        user_out=None,
        item_out=None,
        user_pop=None,
        item_pop=None,
        user_niche=None,
        item_niche=None,
        alpha=1.0,
        beta=1.0,
    ):
        if mode == 'residual':
            user_out = self._require(user_out, 'user_out')
            item_out = self._require(item_out, 'item_out')
            return torch.sum(user_out * item_out, dim=-1)

        if mode == 'two_stream':
            user_pop = self._require(user_pop, 'user_pop')
            item_pop = self._require(item_pop, 'item_pop')
            user_niche = self._require(user_niche, 'user_niche')
            item_niche = self._require(item_niche, 'item_niche')
            pop_score = torch.sum(user_pop * item_pop, dim=-1)
            niche_score = torch.sum(user_niche * item_niche, dim=-1)
            return alpha * pop_score + beta * niche_score

        raise ValueError(f'Unknown score mode: {mode}')


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 4
    num_items = 7
    embed_dim = 12

    user_out = torch.randn(batch_size, embed_dim)
    item_out = torch.randn(num_items, embed_dim)
    user_pop = torch.randn(batch_size, embed_dim)
    user_niche = torch.randn(batch_size, embed_dim)
    item_pop = torch.randn(num_items, embed_dim)
    item_niche = torch.randn(num_items, embed_dim)

    scorer = TwoStreamScorer()

    residual_scores = scorer.score_full(
        mode='residual',
        user_out=user_out,
        item_out=item_out,
    )
    two_stream_scores = scorer.score_full(
        mode='two_stream',
        user_pop=user_pop,
        item_pop=item_pop,
        user_niche=user_niche,
        item_niche=item_niche,
        alpha=1.0,
        beta=1.0,
    )

    assert residual_scores.shape == (batch_size, num_items)
    assert two_stream_scores.shape == (batch_size, num_items)
    print('TwoStreamScorer smoke test passed.')
