# coding: utf-8
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ''):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.abstract_recommender import GeneralRecommender
from models.fitmm import GCN, FrequencyDecompositionModule
from models.pop_niche_splitter import SoftPopularNicheSplitter
from models.two_stream_scorer import TwoStreamScorer


def cfg(config, key, default):
    try:
        return config[key]
    except Exception:
        return default


class SPIN(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items

        self.batch_size = config['train_batch_size']
        self.embed_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.dim_latent = 64
        self.num_gcn_layers = config['num_layers']
        self.num_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.aggr_mode = config['aggr_mode']
        self.reg_weight = config['reg_weight']
        self.ib_weight = config['ib_weight']
        self.drop_rate = 0.1
        self.num_freq_bands = config['num_freq_bands']
        self.ib_direction = config['ib_direction']
        self.ib_alpha = config['ib_alpha']
        self.ib_mu = config['ib_mu']
        self.ib_phi_plus = config['ib_phi_plus']
        self.construction = 'cat'

        self.spin_enable_dual_stream = bool(cfg(config, 'spin_enable_dual_stream', True))
        self.spin_dual_weight = float(cfg(config, 'spin_dual_weight', 0.05))
        self.spin_router_hidden_dim = int(cfg(config, 'spin_router_hidden_dim', 128))
        self.spin_use_activity_popularity = bool(cfg(config, 'spin_use_activity_popularity', True))
        self.spin_score_mode = str(cfg(config, 'spin_score_mode', 'residual')).lower()
        self.spin_pop_alpha = float(cfg(config, 'spin_pop_alpha', 1.0))
        self.spin_niche_beta = float(cfg(config, 'spin_niche_beta', 1.0))
        self.spin_orth_weight = float(cfg(config, 'spin_orth_weight', 0.0))
        self.spin_router_dropout = float(cfg(config, 'spin_router_dropout', 0.0))
        if self.spin_score_mode not in ('residual', 'two_stream'):
            raise ValueError(f'Unsupported spin_score_mode: {self.spin_score_mode}')

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.user_id_embedding = nn.Embedding(self.num_user, self.dim_latent)
        self.item_id_embedding = nn.Embedding(self.num_item, self.dim_latent)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self._build_modal_embeddings()
        self._build_item_graph(dataset_path)
        self._build_interaction_edges()
        self._build_encoders()
        self._build_fusion()
        self._build_splitters()
        self._build_activity_popularity_prior()

        self.last_user_pop = None
        self.last_user_niche = None
        self.last_item_pop = None
        self.last_item_niche = None
        self.last_user_route = None
        self.last_item_route = None

    def _build_modal_embeddings(self):
        self.image_embedding = None
        self.text_embedding = None

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

    def _build_item_graph(self, dataset_path):
        self.use_item_graph = True
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_{self.knn_k}.pt')

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
            return

        adjs = []
        if self.image_embedding is not None:
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            adjs.append((self.mm_image_weight, image_adj))

        if self.text_embedding is not None:
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            text_weight = 1.0 if self.image_embedding is None else (1.0 - self.mm_image_weight)
            adjs.append((text_weight, text_adj))

        if not adjs:
            self.mm_adj = None
            self.use_item_graph = False
            return

        weight, adj = adjs[0]
        self.mm_adj = weight * adj
        for weight, adj in adjs[1:]:
            self.mm_adj = self.mm_adj + weight * adj
        self.mm_adj = self.mm_adj.coalesce()
        torch.save(self.mm_adj, mm_adj_file)

    def _build_interaction_edges(self):
        inter_mat = self.dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = np.column_stack((inter_mat.row, inter_mat.col + self.n_users))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

    def _build_encoders(self):
        self.id_gcn = GCN(
            num_user=self.num_user,
            num_item=self.num_item,
            aggr_mode=self.aggr_mode,
            num_layer=self.num_gcn_layers,
            dim_latent=None,
            device=self.device,
            features=self.item_id_embedding.weight,
        )

        self.v_gcn = None
        if self.v_feat is not None:
            self.v_gcn = GCN(
                num_user=self.num_user,
                num_item=self.num_item,
                aggr_mode=self.aggr_mode,
                num_layer=self.num_gcn_layers,
                dim_latent=self.dim_latent,
                device=self.device,
                features=self.v_feat,
            )

        self.t_gcn = None
        if self.t_feat is not None:
            self.t_gcn = GCN(
                num_user=self.num_user,
                num_item=self.num_item,
                aggr_mode=self.aggr_mode,
                num_layer=self.num_gcn_layers,
                dim_latent=self.dim_latent,
                device=self.device,
                features=self.t_feat,
            )

    def _build_fusion(self):
        self.freq_module = FrequencyDecompositionModule(
            dim_latent=self.dim_latent,
            num_bands=self.num_freq_bands,
            ib_direction=self.ib_direction,
            ib_alpha=self.ib_alpha,
            ib_mu=self.ib_mu,
            ib_phi_plus=self.ib_phi_plus,
        )

    def _build_splitters(self):
        all_dim = 3 * self.dim_latent
        self.user_splitter = SoftPopularNicheSplitter(
            num_bands=self.num_freq_bands,
            embed_dim=all_dim,
            hidden_dim=self.spin_router_hidden_dim,
            dropout=self.spin_router_dropout,
        )
        self.item_splitter = SoftPopularNicheSplitter(
            num_bands=self.num_freq_bands,
            embed_dim=all_dim,
            hidden_dim=self.spin_router_hidden_dim,
            dropout=self.spin_router_dropout,
        )
        self.user_stream_gate = nn.Linear(all_dim, 2)
        self.item_stream_gate = nn.Linear(all_dim, 2)
        self.two_stream_scorer = TwoStreamScorer()

        nn.init.xavier_uniform_(self.user_stream_gate.weight)
        nn.init.zeros_(self.user_stream_gate.bias)
        nn.init.xavier_uniform_(self.item_stream_gate.weight)
        nn.init.zeros_(self.item_stream_gate.bias)

    def _build_activity_popularity_prior(self):
        inter_mat = self.dataset.inter_matrix(form='coo').astype(np.float32)
        user_degree = np.bincount(inter_mat.row, minlength=self.num_user).astype(np.float32)
        item_degree = np.bincount(inter_mat.col, minlength=self.num_item).astype(np.float32)

        self.register_buffer('user_activity', self._degree_to_prior(user_degree))
        self.register_buffer('item_popularity', self._degree_to_prior(item_degree))

    @staticmethod
    def _degree_to_prior(degree_array):
        values = torch.from_numpy(degree_array).float().view(-1, 1)
        values = torch.log1p(values)
        min_value = values.min()
        max_value = values.max()
        if torch.isclose(max_value, min_value):
            return torch.zeros_like(values)
        return (values - min_value) / (max_value - min_value + 1e-12)

    @staticmethod
    def compute_normalized_laplacian(indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=-1).to_dense()
        inv_sqrt = row_sum.pow(-0.5)
        values = inv_sqrt[indices[0]] * inv_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def get_knn_adj_mat(self, embeddings):
        normed = F.normalize(embeddings, p=2, dim=-1)
        sim = normed @ normed.t()
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()

        row = torch.arange(knn_ind.size(0), device=self.device).unsqueeze(1).expand_as(knn_ind)
        indices = torch.stack((row.reshape(-1), knn_ind.reshape(-1)), dim=0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    @staticmethod
    def _propagate_sparse(adj, x, num_layers):
        out = x
        for _ in range(num_layers):
            out = torch.sparse.mm(adj, out)
        return out

    def _zero_node_rep(self):
        return torch.zeros(self.num_user + self.num_item, self.dim_latent, device=self.device)

    def _encode_modalities(self):
        id_rep, _ = self.id_gcn(self.edge_index, self.item_id_embedding.weight)

        if self.v_gcn is not None:
            v_rep, _ = self.v_gcn(self.edge_index, self.image_embedding.weight)
        else:
            v_rep = self._zero_node_rep()

        if self.t_gcn is not None:
            t_rep, _ = self.t_gcn(self.edge_index, self.text_embedding.weight)
        else:
            t_rep = self._zero_node_rep()

        return id_rep, v_rep, t_rep

    @staticmethod
    def _orthogonality_loss(pop_repr, niche_repr):
        cosine = F.cosine_similarity(pop_repr, niche_repr, dim=-1, eps=1e-8)
        return torch.mean(cosine.pow(2))

    def forward(self):
        id_rep, v_rep, t_rep = self._encode_modalities()

        all_rep = torch.cat([id_rep, v_rep, t_rep], dim=1)
        user_rep = all_rep[:self.num_user]
        item_rep = all_rep[self.num_user:]

        item_graph_rep = item_rep
        if self.use_item_graph and self.mm_adj is not None:
            item_graph_rep = self._propagate_sparse(self.mm_adj, item_rep, self.num_mm_layers)

        user_bands = self.freq_module.frequency_decompose_svd_separate(user_rep)
        item_bands = self.freq_module.frequency_decompose_svd_separate(item_rep)
        item_graph_bands = self.freq_module.frequency_decompose_svd_separate(item_graph_rep)
        fused_item_bands = [a + b for a, b in zip(item_bands, item_graph_bands)]

        h_user_fitmm, user_ib = self.freq_module.user_fusion(user_bands, user_rep)
        h_item_fitmm, item_ib = self.freq_module.item_fusion(fused_item_bands, item_rep)

        user_prior = self.user_activity if self.spin_use_activity_popularity else None
        item_prior = self.item_popularity if self.spin_use_activity_popularity else None

        u_pop, u_niche, user_route = self.user_splitter(user_bands, user_rep, user_prior)
        i_pop, i_niche, item_route = self.item_splitter(fused_item_bands, item_rep, item_prior)

        stream_gate_u = F.softmax(self.user_stream_gate(user_rep), dim=-1)
        stream_gate_i = F.softmax(self.item_stream_gate(item_rep), dim=-1)
        h_user_dual = stream_gate_u[:, 0:1] * u_pop + stream_gate_u[:, 1:2] * u_niche
        h_item_dual = stream_gate_i[:, 0:1] * i_pop + stream_gate_i[:, 1:2] * i_niche

        if self.spin_enable_dual_stream:
            user_out = h_user_fitmm + self.spin_dual_weight * h_user_dual
            item_out = h_item_fitmm + self.spin_dual_weight * h_item_dual
        else:
            user_out = h_user_fitmm
            item_out = h_item_fitmm

        aux_loss = user_ib + item_ib
        if self.spin_orth_weight > 0:
            orth_loss = self._orthogonality_loss(u_pop, u_niche) + self._orthogonality_loss(i_pop, i_niche)
            aux_loss = aux_loss + self.spin_orth_weight * orth_loss

        self.last_user_pop = u_pop
        self.last_user_niche = u_niche
        self.last_item_pop = i_pop
        self.last_item_niche = i_niche
        self.last_user_route = user_route
        self.last_item_route = item_route

        return user_out, item_out, aux_loss

    @staticmethod
    def pairwise_bce_loss(user_emb, pos_item_emb, neg_item_emb):
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def calculate_loss(self, interaction):
        user_idx, pos_idx, neg_idx = interaction[0], interaction[1], interaction[2]
        user_emb_all, item_emb_all, aux_loss = self.forward()

        user_emb = user_emb_all[user_idx]
        pos_item_emb = item_emb_all[pos_idx]
        neg_item_emb = item_emb_all[neg_idx]

        rec_loss = self.pairwise_bce_loss(user_emb, pos_item_emb, neg_item_emb)
        return rec_loss + self.ib_weight * aux_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_out, item_out, _ = self.forward()

        if self.spin_score_mode == 'residual':
            return self.two_stream_scorer.score_full(
                mode='residual',
                user_out=user_out[user],
                item_out=item_out,
            )

        if self.spin_score_mode == 'two_stream':
            return self.two_stream_scorer.score_full(
                mode='two_stream',
                user_pop=self.last_user_pop[user],
                item_pop=self.last_item_pop,
                user_niche=self.last_user_niche[user],
                item_niche=self.last_item_niche,
                alpha=self.spin_pop_alpha,
                beta=self.spin_niche_beta,
            )

        raise ValueError(f'Unknown spin_score_mode: {self.spin_score_mode}')


if __name__ == '__main__':
    splitter = SoftPopularNicheSplitter(num_bands=3, embed_dim=12, hidden_dim=8, dropout=0.0)
    scorer = TwoStreamScorer()
    print(f'SoftPopularNicheSplitter import smoke test: {splitter.__class__.__name__}')
    print(f'TwoStreamScorer import smoke test: {scorer.__class__.__name__}')
    print('SPIN module import smoke test passed. Full forward requires the project training entrypoint with a real RecDataset/TrainDataLoader.')
