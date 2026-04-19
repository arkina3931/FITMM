# coding: utf-8
import torch


class PrototypeScorer(object):
    @staticmethod
    def _pair_score(left_emb, right_emb):
        return torch.sum(left_emb * right_emb, dim=-1)

    def score_pair(
        self,
        base_pos_score,
        base_neg_score,
        z_pos_pop,
        z_pos_niche,
        z_neg_pop,
        z_neg_niche,
        i_pop_pos,
        i_niche_pos,
        i_pop_neg,
        i_niche_neg,
        pos_score_weight,
        anti_score_weight,
        use_pos_score,
        use_anti_score,
    ):
        pos_score = base_pos_score
        neg_score = base_neg_score

        if use_pos_score:
            pos_pos_score = self._pair_score(z_pos_pop, i_pop_pos) + self._pair_score(z_pos_niche, i_niche_pos)
            neg_pos_score = self._pair_score(z_pos_pop, i_pop_neg) + self._pair_score(z_pos_niche, i_niche_neg)
            pos_score = pos_score + pos_score_weight * pos_pos_score
            neg_score = neg_score + pos_score_weight * neg_pos_score

        if use_anti_score and z_neg_pop is not None and z_neg_niche is not None:
            pos_anti_score = self._pair_score(z_neg_pop, i_pop_pos) + self._pair_score(z_neg_niche, i_niche_pos)
            neg_anti_score = self._pair_score(z_neg_pop, i_pop_neg) + self._pair_score(z_neg_niche, i_niche_neg)
            pos_score = pos_score - anti_score_weight * pos_anti_score
            neg_score = neg_score - anti_score_weight * neg_anti_score

        return pos_score, neg_score

    def score_full(
        self,
        base_scores,
        z_pos_pop,
        z_pos_niche,
        z_neg_pop,
        z_neg_niche,
        i_pop_all,
        i_niche_all,
        pos_score_weight,
        anti_score_weight,
        use_pos_score,
        use_anti_score,
        pop_alpha=1.0,
        niche_beta=1.0,
    ):
        scores = base_scores
        if use_pos_score:
            pos_scores = pop_alpha * (z_pos_pop @ i_pop_all.t()) + niche_beta * (z_pos_niche @ i_niche_all.t())
            scores = scores + pos_score_weight * pos_scores
        if use_anti_score and z_neg_pop is not None and z_neg_niche is not None:
            anti_scores = pop_alpha * (z_neg_pop @ i_pop_all.t()) + niche_beta * (z_neg_niche @ i_niche_all.t())
            scores = scores - anti_score_weight * anti_scores
        return scores


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 4
    num_items = 7
    embed_dim = 12
    scorer = PrototypeScorer()

    base_pos = torch.randn(batch_size)
    base_neg = torch.randn(batch_size)
    base_full = torch.randn(batch_size, num_items)
    z_pos_pop = torch.randn(batch_size, embed_dim)
    z_pos_niche = torch.randn(batch_size, embed_dim)
    z_neg_pop = torch.randn(batch_size, embed_dim)
    z_neg_niche = torch.randn(batch_size, embed_dim)
    i_pop_pos = torch.randn(batch_size, embed_dim)
    i_niche_pos = torch.randn(batch_size, embed_dim)
    i_pop_neg = torch.randn(batch_size, embed_dim)
    i_niche_neg = torch.randn(batch_size, embed_dim)
    i_pop_all = torch.randn(num_items, embed_dim)
    i_niche_all = torch.randn(num_items, embed_dim)

    pair_pos, pair_neg = scorer.score_pair(
        base_pos_score=base_pos,
        base_neg_score=base_neg,
        z_pos_pop=z_pos_pop,
        z_pos_niche=z_pos_niche,
        z_neg_pop=z_neg_pop,
        z_neg_niche=z_neg_niche,
        i_pop_pos=i_pop_pos,
        i_niche_pos=i_niche_pos,
        i_pop_neg=i_pop_neg,
        i_niche_neg=i_niche_neg,
        pos_score_weight=0.01,
        anti_score_weight=0.001,
        use_pos_score=True,
        use_anti_score=True,
    )
    full_scores = scorer.score_full(
        base_scores=base_full,
        z_pos_pop=z_pos_pop,
        z_pos_niche=z_pos_niche,
        z_neg_pop=z_neg_pop,
        z_neg_niche=z_neg_niche,
        i_pop_all=i_pop_all,
        i_niche_all=i_niche_all,
        pos_score_weight=0.01,
        anti_score_weight=0.001,
        use_pos_score=True,
        use_anti_score=True,
        pop_alpha=1.0,
        niche_beta=1.0,
    )

    assert pair_pos.shape == (batch_size,)
    assert pair_neg.shape == (batch_size,)
    assert full_scores.shape == (batch_size, num_items)
    print('PrototypeScorer smoke test passed.')
