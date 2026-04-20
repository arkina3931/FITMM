# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import dirichlet

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from common.loss import BPRLoss, EmbLoss
from pytorch_wavelets import DWT1D  # type: ignore

class _PopularNicheGraphBuilder(nn.Module):
    """Build user/item popular-vs-niche weights from a sparse interaction matrix."""

    def __init__(self, interaction_matrix, wavelet="db3", level=3, device="cuda:0"):
        super().__init__()
        self.device = device
        self.dtype = torch.float32
        self.wavelet = wavelet
        self.level = level

        self.R = interaction_matrix.coalesce().to(device=device, dtype=self.dtype)
        self.num_users = self.R.size(0)
        self.num_items = self.R.size(1)
        self.indices = self.R.indices()
        self.values = self.R.values()

        self.dwt = DWT1D(wave=wavelet, J=level).to(device)

        self.item_popularity = None
        self.user_activity = None

    def _compute_item_popularity(self):
        if self.item_popularity is None:
            pop = torch.zeros(self.num_items, device=self.device, dtype=self.dtype)
            counts = torch.bincount(self.indices[1], minlength=self.num_items)
            log_counts = torch.log1p(counts.to(self.dtype))
            pop.scatter_add_(0, self.indices[1], self.values * log_counts[self.indices[1]])
            self.item_popularity = pop / (torch.norm(pop) + 1e-8)
        return self.item_popularity

    def _compute_user_activity(self):
        if self.user_activity is None:
            act = torch.zeros(self.num_users, device=self.device, dtype=self.dtype)
            pop = self._compute_item_popularity()
            pop_vals = pop[self.indices[1]] + 1e-8
            act.scatter_add_(0, self.indices[0], self.values / torch.log1p(pop_vals))
            self.user_activity = act / (torch.norm(act) + 1e-8)
        return self.user_activity

    def _get_user_signal(self, user_idx):
        mask = self.indices[0] == user_idx
        item_ids = self.indices[1][mask]
        vals = self.values[mask]

        signal = torch.zeros(self.num_items, device=self.device, dtype=self.dtype)
        signal[item_ids] = vals * (1.0 + self.user_activity[user_idx])
        return signal.unsqueeze(0).unsqueeze(0)

    def _get_item_signal(self, item_idx):
        mask = self.indices[1] == item_idx
        user_ids = self.indices[0][mask]
        vals = self.values[mask]

        signal = torch.zeros(self.num_users, device=self.device, dtype=self.dtype)
        signal[user_ids] = vals * (1.0 + self.item_popularity[item_idx])
        return signal.unsqueeze(0).unsqueeze(0)

    def _wavelet_transform_batch(self, is_user=True, batch_size=32):
        num_entities = self.num_users if is_user else self.num_items
        low_freq = torch.zeros(num_entities, device=self.device, dtype=self.dtype)
        high_freq = torch.zeros(num_entities, device=self.device, dtype=self.dtype)

        for start in range(0, num_entities, batch_size):
            end = min(start + batch_size, num_entities)
            batch_low = []
            batch_high = []

            for idx in range(start, end):
                signal = self._get_user_signal(idx) if is_user else self._get_item_signal(idx)
                low_band, high_bands = self.dwt(signal)

                batch_low.append(float(torch.mean(low_band).item()))
                high_energy = torch.sum(torch.stack([torch.norm(high_band) for high_band in high_bands])).item()
                batch_high.append(float(high_energy))

            low_freq[start:end] = torch.tensor(batch_low, device=self.device, dtype=self.dtype)
            high_freq[start:end] = torch.tensor(batch_high, device=self.device, dtype=self.dtype)

        return low_freq, high_freq

    def _dirichlet_weight(self, low, high):
        alpha_low = torch.clamp(low, min=1e-6)
        alpha_high = torch.clamp(high, min=1e-6)
        alpha_sum = alpha_low + alpha_high
        alpha_low = alpha_low / alpha_sum
        alpha_high = alpha_high / alpha_sum

        weights = []
        for a1, a2 in zip(alpha_low.detach().cpu().numpy(), alpha_high.detach().cpu().numpy()):
            weights.append(dirichlet.rvs([a1 * 10.0, a2 * 10.0], size=1)[0, 0])

        return torch.tensor(weights, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def build_weights(self, batch_size=32):
        self._compute_item_popularity()
        self._compute_user_activity()

        user_low, user_high = self._wavelet_transform_batch(is_user=True, batch_size=batch_size)
        item_low, item_high = self._wavelet_transform_batch(is_user=False, batch_size=batch_size)

        user_pop_weight = self._dirichlet_weight(user_low, user_high)
        item_pop_weight = self._dirichlet_weight(item_low, item_high)

        return {
            "user_pop_weight": user_pop_weight,
            "user_niche_weight": 1.0 - user_pop_weight,
            "item_pop_weight": item_pop_weight,
            "item_niche_weight": 1.0 - item_pop_weight,
            "user_activity": self.user_activity,
            "item_popularity": self.item_popularity,
        }


class PNG(GeneralRecommender):
    """Popular-Niche Graph recommender with BPR training."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.dataset = dataset
        self.embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.wavelet = 'db3'
        self.level = config['level']
        self.graph_batch_size = config['graph_batch_size']
        self.png_pop_scale = config['png_pop_scale']
        self.png_niche_scale = config['png_niche_scale']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.user_pop_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_niche_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_pop_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_niche_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        interaction_matrix = self._build_interaction_matrix(dataset)
        self.register_buffer("interaction_matrix", interaction_matrix)

        graph_builder = _PopularNicheGraphBuilder(
            interaction_matrix=interaction_matrix,
            wavelet=self.wavelet,
            level=self.level,
            device=self.device,
        )
        graph_stats = graph_builder.build_weights(batch_size=self.graph_batch_size)

        self.register_buffer("user_pop_weight", graph_stats["user_pop_weight"])
        self.register_buffer("user_niche_weight", graph_stats["user_niche_weight"])
        self.register_buffer("item_pop_weight", graph_stats["item_pop_weight"])
        self.register_buffer("item_niche_weight", graph_stats["item_niche_weight"])
        self.register_buffer("user_activity", graph_stats["user_activity"])
        self.register_buffer("item_popularity", graph_stats["item_popularity"])

        self.apply(xavier_normal_initialization)

    def _build_interaction_matrix(self, dataset):
        inter_mat = dataset.inter_matrix(form="coo").astype(np.float32)
        indices = torch.from_numpy(np.vstack((inter_mat.row, inter_mat.col)).astype(np.int64))
        values = torch.from_numpy(inter_mat.data.astype(np.float32))
        return torch.sparse_coo_tensor(
            indices,
            values,
            torch.Size(inter_mat.shape),
            device=self.device,
            dtype=torch.float32,
        ).coalesce()

    def forward(self):
        user_base = self.user_embedding.weight
        item_base = self.item_embedding.weight

        user_pop = self.user_pop_embedding.weight
        user_niche = self.user_niche_embedding.weight
        item_pop = self.item_pop_embedding.weight
        item_niche = self.item_niche_embedding.weight

        user_embedding = (
            user_base
            + self.png_pop_scale * self.user_pop_weight.unsqueeze(1) * user_pop
            + self.png_niche_scale * self.user_niche_weight.unsqueeze(1) * user_niche
        )
        item_embedding = (
            item_base
            + self.png_pop_scale * self.item_pop_weight.unsqueeze(1) * item_pop
            + self.png_niche_scale * self.item_niche_weight.unsqueeze(1) * item_niche
        )
        return user_embedding, item_embedding

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user]
        pos_e = item_embeddings[pos_item]
        neg_e = item_embeddings[neg_item]

        pos_scores = torch.mul(user_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(user_e, neg_e).sum(dim=1)

        mf_loss = self.loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user]
        return torch.matmul(user_e, item_embeddings.transpose(0, 1))
