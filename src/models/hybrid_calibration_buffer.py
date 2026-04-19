# coding: utf-8
import math

import torch


class LightweightHybridCalibrationBuffer(object):
    def __init__(
        self,
        item_popularity,
        pop_buffer_size=500,
        niche_buffer_size=500,
        calib_candidate_size=32,
        calib_topk=3,
        calib_popularity_weight=0.1,
        calib_niche_pop_penalty=0.1,
        niche_buffer_mode='exclude_top_popular',
        niche_exclude_top_ratio=0.2,
    ):
        self.item_popularity = item_popularity.view(-1)
        self.device = self.item_popularity.device
        self.num_items = self.item_popularity.numel()
        self.pop_buffer_size = int(min(pop_buffer_size, self.num_items))
        self.niche_buffer_size = int(min(niche_buffer_size, self.num_items))
        self.calib_candidate_size = int(max(1, calib_candidate_size))
        self.calib_topk = int(max(1, calib_topk))
        self.calib_popularity_weight = float(calib_popularity_weight)
        self.calib_niche_pop_penalty = float(calib_niche_pop_penalty)
        self.niche_buffer_mode = str(niche_buffer_mode).lower()
        self.niche_exclude_top_ratio = float(niche_exclude_top_ratio)

        self.sorted_items = torch.argsort(self.item_popularity, descending=True)
        self.popular_item_buffer = self.sorted_items[:self.pop_buffer_size]
        self.niche_pool = self._build_niche_pool()
        self.niche_item_buffer = self._sample_from_pool(self.niche_pool, self.niche_buffer_size)

    def _build_niche_pool(self):
        if self.niche_buffer_mode != 'exclude_top_popular':
            raise ValueError(f'Unsupported niche_buffer_mode: {self.niche_buffer_mode}')
        exclude_count = int(math.ceil(self.num_items * self.niche_exclude_top_ratio))
        exclude_count = min(max(exclude_count, 0), self.num_items)
        niche_pool = self.sorted_items[exclude_count:]
        if niche_pool.numel() == 0:
            return self.sorted_items
        return niche_pool

    def _sample_from_pool(self, pool, sample_size):
        if pool.numel() <= sample_size:
            return pool
        perm = torch.randperm(pool.numel(), device=pool.device)[:sample_size]
        return pool[perm]

    def refresh_buffers(self):
        self.popular_item_buffer = self.sorted_items[:self.pop_buffer_size]
        self.niche_item_buffer = self._sample_from_pool(self.niche_pool, self.niche_buffer_size)

    def _sample_candidates(self, buffer_tensor):
        return self._sample_from_pool(buffer_tensor, min(self.calib_candidate_size, buffer_tensor.numel()))

    def build_calibration_targets(
        self,
        user_idx,
        pos_idx,
        user_out_batch,
        u_niche_batch,
        i_pop_all,
        i_niche_all,
        item_out_all,
        item_popularity,
    ):
        batch_size = user_idx.size(0)
        target_device = item_out_all.device
        pop_candidates = self._sample_candidates(self.popular_item_buffer).to(target_device)
        niche_candidates = self._sample_candidates(self.niche_item_buffer).to(target_device)
        pos_idx = pos_idx.to(target_device)

        pop_item_out = item_out_all[pop_candidates]  # [Cp, D]
        pop_popularity = item_popularity[pop_candidates].view(1, -1)  # [1, Cp]
        pop_scores = user_out_batch @ pop_item_out.t()  # [B, Cp]
        pop_scores = pop_scores + self.calib_popularity_weight * pop_popularity
        pop_topk = min(self.calib_topk, pop_scores.size(1))
        pop_topk_idx = torch.topk(pop_scores, k=pop_topk, dim=-1).indices  # [B, K]
        pop_selected = pop_candidates[pop_topk_idx]  # [B, K]
        target_calib_pop = i_pop_all[pop_selected].mean(dim=1)  # [B, D]

        niche_item = i_niche_all[niche_candidates]  # [Cn, D]
        niche_pos = i_niche_all[pos_idx]  # [B, D]
        score_niche_pos = niche_pos @ niche_item.t()  # [B, Cn]
        score_niche_user = u_niche_batch @ niche_item.t()  # [B, Cn]
        niche_popularity = item_popularity[niche_candidates].view(1, -1)  # [1, Cn]
        niche_scores = score_niche_pos + score_niche_user - self.calib_niche_pop_penalty * niche_popularity
        niche_topk = min(self.calib_topk, niche_scores.size(1))
        niche_topk_idx = torch.topk(niche_scores, k=niche_topk, dim=-1).indices  # [B, K]
        niche_selected = niche_candidates[niche_topk_idx]  # [B, K]
        target_calib_niche = i_niche_all[niche_selected].mean(dim=1)  # [B, D]

        assert target_calib_pop.shape[0] == batch_size
        assert target_calib_niche.shape[0] == batch_size
        return target_calib_pop, target_calib_niche


if __name__ == '__main__':
    torch.manual_seed(2026)

    num_items = 50
    batch_size = 4
    embed_dim = 12

    popularity = torch.linspace(1.0, 0.0, num_items)
    buffer = LightweightHybridCalibrationBuffer(
        item_popularity=popularity,
        pop_buffer_size=10,
        niche_buffer_size=12,
        calib_candidate_size=6,
        calib_topk=3,
        calib_popularity_weight=0.1,
        calib_niche_pop_penalty=0.1,
        niche_buffer_mode='exclude_top_popular',
        niche_exclude_top_ratio=0.2,
    )
    buffer.refresh_buffers()

    user_idx = torch.arange(batch_size)
    pos_idx = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    user_out_batch = torch.randn(batch_size, embed_dim)
    u_niche_batch = torch.randn(batch_size, embed_dim)
    i_pop_all = torch.randn(num_items, embed_dim)
    i_niche_all = torch.randn(num_items, embed_dim)
    item_out_all = torch.randn(num_items, embed_dim)

    target_pop, target_niche = buffer.build_calibration_targets(
        user_idx=user_idx,
        pos_idx=pos_idx,
        user_out_batch=user_out_batch,
        u_niche_batch=u_niche_batch,
        i_pop_all=i_pop_all,
        i_niche_all=i_niche_all,
        item_out_all=item_out_all,
        item_popularity=popularity,
    )

    assert target_pop.shape == (batch_size, embed_dim)
    assert target_niche.shape == (batch_size, embed_dim)
    print('LightweightHybridCalibrationBuffer smoke test passed.')
