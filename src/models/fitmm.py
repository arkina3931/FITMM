# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, remove_self_loops

from common.abstract_recommender import GeneralRecommender


class FITMM(GeneralRecommender):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items

        # config
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

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
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

        self.mm_adj = sum(w * adj for w, adj in adjs)
        torch.save(self.mm_adj, mm_adj_file)

    def _build_interaction_edges(self):
        # packing interaction in training into edge_index
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

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def get_knn_adj_mat(self, embeddings):
        normed = F.normalize(embeddings, p=2, dim=-1)
        sim = normed @ normed.t()
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()

        # construct sparse adj
        row = torch.arange(knn_ind.size(0), device=self.device).unsqueeze(1).expand_as(knn_ind)
        indices = torch.stack((row.reshape(-1), knn_ind.reshape(-1)), dim=0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    @staticmethod
    def compute_normalized_laplacian(indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=-1).to_dense()
        inv_sqrt = row_sum.pow(-0.5)
        values = inv_sqrt[indices[0]] * inv_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, values, adj_size)

    @staticmethod
    def _propagate_sparse(adj, x, num_layers):
        out = x
        for _ in range(num_layers):
            out = torch.sparse.mm(adj, out)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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

    def forward(self):
        id_rep, v_rep, t_rep = self._encode_modalities()

        all_rep = torch.cat([id_rep, v_rep, t_rep], dim=1)
        user_rep = all_rep[:self.num_user]
        item_rep = all_rep[self.num_user:]

        # ---- Item-side graph propagation ----
        item_graph_rep = item_rep
        if self.use_item_graph and self.mm_adj is not None:
            item_graph_rep = self._propagate_sparse(self.mm_adj, item_rep, self.num_mm_layers)

        # ---- Frequency-aware Decomposition fusion ----
        user_out, item_out, ib_loss = self.freq_module(user_rep, item_rep, item_graph_rep)
        return user_out, item_out, ib_loss

    # ------------------------------------------------------------------
    # Loss / inference
    # ------------------------------------------------------------------
    @staticmethod
    def pairwise_bce_loss(user_emb, pos_item_emb, neg_item_emb):
        """
        Treat (user, pos_item) as positive samples and (user, neg_item) as negative samples,
        then concatenate them and optimize using a binary cross-entropy loss.
        """
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def calculate_loss(self, interaction):
        user_idx, pos_idx, neg_idx = interaction[0], interaction[1], interaction[2]
        user_emb_all, item_emb_all, ib_loss = self.forward()

        user_emb = user_emb_all[user_idx]
        pos_item_emb = item_emb_all[pos_idx]
        neg_item_emb = item_emb_all[neg_idx]

        rec_loss = self.pairwise_bce_loss(user_emb, pos_item_emb, neg_item_emb)
        return rec_loss + self.ib_weight * ib_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _ = self.forward()
        return user_emb[user] @ item_emb.t()


class GCN(nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, num_layer, dim_latent=None, device=None, features=None):
        super().__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device

        pref_dim = self.dim_latent or self.dim_feat
        self.preference = nn.Parameter(torch.randn(num_user, pref_dim, device=device))
        nn.init.xavier_normal_(self.preference)

        self.proj = None
        if self.dim_latent is not None:
            self.proj = nn.Sequential(
                nn.Linear(self.dim_feat, 4 * self.dim_latent),
                nn.LeakyReLU(),
                nn.Linear(4 * self.dim_latent, self.dim_latent),
            )

        self.conv = BaseGCN(aggr=aggr_mode)

    def forward(self, edge_index, features):
        item_feat = self.proj(features) if self.proj is not None else features
        x = torch.cat((self.preference, item_feat), dim=0)
        x = F.normalize(x, dim=-1)

        outs = [x]
        h = x
        for _ in range(self.num_layer):
            h = self.conv(h, edge_index)
            outs.append(h)
        return sum(outs), self.preference


class BaseGCN(MessagePassing):
    def __init__(self, aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr != 'add':
            return x_j
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j


class FrequencyDecompositionModule(nn.Module):
    def __init__(self, dim_latent, num_bands, ib_direction, ib_alpha, ib_mu, ib_phi_plus):
        """
        Frequency decomposition + MoE structure:
        - M: number of frequency bands
        """
        super().__init__()
        self.M = num_bands
        self.dim_latent = dim_latent
        self.all_dim = 3 * dim_latent
        self.user_fusion = TaskAwareFrequencyFusion(num_bands, self.all_dim, ib_direction, ib_alpha, ib_mu, ib_phi_plus)
        self.item_fusion = TaskAwareFrequencyFusion(num_bands, self.all_dim, ib_direction, ib_alpha, ib_mu, ib_phi_plus)

    def frequency_decompose_svd(self, rep):
        """
        Perform SVD on `rep` and split (U, S, Vh) into M parts according to the singular values.
        Finally, return M partial-sum matrices, each having the same shape as `rep`.
        """
        M = self.M
        # rep: [N, F]
        U, S, Vh = torch.linalg.svd(rep, full_matrices=False)
        # U:  [N, F]
        # S:  [F, ]  (one dimension)
        # Vh: [F, F]

        N, F_ = U.shape  # F_ should be as F
        assert F_ == S.shape[0] and F_ == Vh.shape[0] == Vh.shape[1], \
            f"Shape mismatch in SVD: U({U.shape}), S({S.shape}), Vh({Vh.shape})"

        # calculate each band size
        # e.g. F=192, M=3, split_sizes = [64,64,64]
        split_sizes = []
        base = F_ // M  # base size for each band
        remainder = F_ % M
        for i in range(M):
            size_i = base + (1 if i < remainder else 0)
            split_sizes.append(size_i)
        # sum(split_sizes) == F_

        # splitting S, U, Vh one by one
        freq_components = []
        start = 0
        for size_i in split_sizes:
            end = start + size_i

            # S_i: [size_i, ]
            S_i = S[start:end]
            # U_i: [N, size_i]
            U_i = U[:, start:end]
            # V_i: [size_i, F]
            V_i = Vh[start:end, :]

            # reconstruct corresponding matrix for the band = sum_{k in [start, end]}(S_k * U_{:,k} outer Vh_{k,:})
            # U_i @ diag(S_i) @ V_i
            diag_S_i = torch.diag(S_i)              # (size_i, size_i)
            partial_rep = U_i @ diag_S_i @ V_i      # (N, F)
            freq_components.append(partial_rep)

            start = end

        return freq_components  # List[M], with each element shape (N, F)

    def frequency_decompose_svd_separate(self, rep):
        """ Split modalities first, then perform SVD decomposition separately for each modality. """
        id_rep, vis_rep, txt_rep = torch.split(rep, [self.dim_latent, self.dim_latent, self.dim_latent], dim=-1)
        id_bands = self.frequency_decompose_svd(id_rep)
        vis_bands = self.frequency_decompose_svd(vis_rep)
        txt_bands = self.frequency_decompose_svd(txt_rep)
        return [torch.cat([id_bands[i], vis_bands[i], txt_bands[i]], dim=-1) for i in range(self.M)]

    def forward(self, user_rep, item_rep, item_graph_rep):
        """
        - user_rep: [num_user, id_dim + v_dim + t_dim], GCN user representations
        - item_rep: [num_item, id_dim + v_dim + t_dim], GCN item representations
        """
        user_bands = self.frequency_decompose_svd_separate(user_rep)
        item_bands = self.frequency_decompose_svd_separate(item_rep)
        item_graph_bands = self.frequency_decompose_svd_separate(item_graph_rep)

        fused_item_bands = [a + b for a, b in zip(item_bands, item_graph_bands)]

        user_out, user_ib = self.user_fusion(user_bands, user_rep)
        item_out, item_ib = self.item_fusion(fused_item_bands, item_rep)
        return user_out, item_out, user_ib + item_ib


class TaskAwareFrequencyFusion(nn.Module):
    def __init__(self, num_bands, embed_dim, ib_direction='Pos', ib_alpha=1.0, ib_mu=1.0, ib_phi_plus=0.0):
        super().__init__()
        self.num_bands = num_bands
        self.ib_direction = ib_direction
        self.ib_alpha = ib_alpha
        self.ib_mu = ib_mu
        self.ib_phi_plus = ib_phi_plus

        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.ones(num_bands))
        self.gate_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Frequency gate
        if ib_direction == 'Dual':
            # we provide a new direction to allow both positive/negative deviations for expanding
            self.freq_gate = nn.Sequential(nn.Linear(embed_dim, num_bands), nn.Tanh())
        elif ib_direction == 'Pos':
            self.freq_gate = nn.Sequential(nn.Linear(embed_dim, num_bands), nn.Sigmoid())
        else:
            raise ValueError(f'Unknown ib_direction: {ib_direction}')

    def ib_surrogate_loss_from_gate(self, gate_values, eps=1e-12, enforce_nonneg=True):
        delta = gate_values - 1.0
        if self.ib_direction == 'Pos' and enforce_nonneg:
            delta_for_threshold = F.relu(delta)
            delta = delta_for_threshold
        elif self.ib_direction == 'Dual':
            # we provide a new direction to allow both positive/negative deviations for expanding
            # adjustment here correspondingly
            delta_for_threshold = delta.abs()
        else:
            delta_for_threshold = delta

        delta_norm_sq = torch.sum(delta * delta, dim=1)
        term1 = self.ib_alpha * delta_norm_sq.mean()

        delta_norm = torch.sqrt(delta_norm_sq + eps)
        exceed = F.relu(delta_for_threshold - self.ib_phi_plus)
        term2 = self.ib_mu * (delta_norm * exceed.sum(dim=1)).mean()
        return term1 + term2

    def forward(self, band_components, task_emb, enforce_nonneg=True):
        """
        Returns:
            fused_emb: (N, D)
            ib_loss:   scalar tensor
        """
        band_tensor = torch.stack(band_components, dim=1)
        band_gates = 1.0 + self.gate_scale * self.freq_gate(task_emb)
        ib_loss = self.ib_surrogate_loss_from_gate(band_gates, enforce_nonneg=enforce_nonneg)

        band_weights = torch.sigmoid(self.freq_weights).view(1, self.num_bands, 1)
        fused = torch.sum(band_weights * band_gates.unsqueeze(-1) * band_tensor, dim=1)
        return fused, ib_loss
