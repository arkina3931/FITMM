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
from losses.anti_prototype_losses import anti_calibration_loss, anti_norm_loss, anti_separation_loss
from losses.prototype_align import prototype_cosine_alignment_loss
from models.anti_prototype import AntiPrototypeGenerator
from models.fitmm import GCN, FrequencyDecompositionModule
from models.hybrid_calibration_buffer import LightweightHybridCalibrationBuffer
from models.pop_niche_splitter import SoftPopularNicheSplitter
from models.prototype_flow import PrototypeFlow
from models.prototype_scorer import PrototypeScorer
from models.two_stream_scorer import TwoStreamScorer


def cfg(config, key, default):
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


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
        self.all_dim = 3 * self.dim_latent

        self.spin_enable_dual_stream = bool(cfg(config, 'spin_enable_dual_stream', True))
        self.spin_dual_weight = float(cfg(config, 'spin_dual_weight', 0.05))
        self.spin_router_hidden_dim = int(cfg(config, 'spin_router_hidden_dim', 128))
        self.spin_use_activity_popularity = bool(cfg(config, 'spin_use_activity_popularity', True))
        self.spin_score_mode = str(cfg(config, 'spin_score_mode', 'residual')).lower()
        self.spin_pop_alpha = float(cfg(config, 'spin_pop_alpha', 1.0))
        self.spin_niche_beta = float(cfg(config, 'spin_niche_beta', 1.0))
        self.spin_orth_weight = float(cfg(config, 'spin_orth_weight', 0.0))
        self.spin_router_dropout = float(cfg(config, 'spin_router_dropout', 0.0))
        self.spin_enable_pos_flow = bool(cfg(config, 'spin_enable_pos_flow', True))
        self.spin_pos_flow_hidden_dim = int(cfg(config, 'spin_pos_flow_hidden_dim', 128))
        self.spin_pos_flow_time_dim = int(cfg(config, 'spin_pos_flow_time_dim', 32))
        self.spin_pos_flow_weight = float(cfg(config, 'spin_pos_flow_weight', 0.01))
        self.spin_pos_align_weight = float(cfg(config, 'spin_pos_align_weight', 0.01))
        self.spin_pos_score_weight = float(cfg(config, 'spin_pos_score_weight', 0.02))
        self.spin_pos_source_noise_std = float(cfg(config, 'spin_pos_source_noise_std', 0.05))
        self.spin_use_pos_score = bool(cfg(config, 'spin_use_pos_score', True))
        self.spin_detach_pos_target = bool(cfg(config, 'spin_detach_pos_target', True))
        self.spin_detach_pos_condition = bool(cfg(config, 'spin_detach_pos_condition', False))
        self.spin_separate_pos_flow_heads = bool(cfg(config, 'spin_separate_pos_flow_heads', True))
        self.spin_proto_normalize = bool(cfg(config, 'spin_proto_normalize', True))
        self.spin_flow_velocity_type = str(cfg(config, 'spin_flow_velocity_type', 'moe')).lower()
        self.spin_moe_enable = bool(cfg(config, 'spin_moe_enable', True))
        self.spin_moe_num_experts = int(cfg(config, 'spin_moe_num_experts', 4))
        self.spin_moe_hidden_dim = int(cfg(config, 'spin_moe_hidden_dim', self.spin_pos_flow_hidden_dim))
        self.spin_moe_router_hidden_dim = int(cfg(config, 'spin_moe_router_hidden_dim', 128))
        self.spin_moe_router_dropout = float(cfg(config, 'spin_moe_router_dropout', 0.0))
        self.spin_moe_expert_dropout = float(cfg(config, 'spin_moe_expert_dropout', 0.0))
        self.spin_moe_dense = bool(cfg(config, 'spin_moe_dense', True))
        self.spin_moe_entropy_weight = float(cfg(config, 'spin_moe_entropy_weight', 0.0))
        self.spin_moe_balance_weight = float(cfg(config, 'spin_moe_balance_weight', 0.0))
        self.spin_enable_anti_proto = bool(cfg(config, 'spin_enable_anti_proto', True))
        self.spin_use_anti_score = bool(cfg(config, 'spin_use_anti_score', True))
        self.spin_anti_eta_pop = float(cfg(config, 'spin_anti_eta_pop', 0.5))
        self.spin_anti_eta_niche = float(cfg(config, 'spin_anti_eta_niche', 0.5))
        self.spin_anti_gate_hidden_dim = int(cfg(config, 'spin_anti_gate_hidden_dim', 128))
        self.spin_anti_gate_type = str(cfg(config, 'spin_anti_gate_type', 'scalar')).lower()
        self.spin_anti_score_weight = float(cfg(config, 'spin_anti_score_weight', 0.001))
        self.spin_anti_sep_weight = float(cfg(config, 'spin_anti_sep_weight', 0.001))
        self.spin_anti_sep_margin = float(cfg(config, 'spin_anti_sep_margin', 0.2))
        self.spin_anti_norm_weight = float(cfg(config, 'spin_anti_norm_weight', 0.0001))
        self.spin_anti_proto_normalize = bool(cfg(config, 'spin_anti_proto_normalize', self.spin_proto_normalize))
        self.spin_enable_hybrid_calibration = bool(cfg(config, 'spin_enable_hybrid_calibration', True))
        self.spin_anti_calib_weight = float(cfg(config, 'spin_anti_calib_weight', 0.001))
        self.spin_hybrid_update_interval = int(cfg(config, 'spin_hybrid_update_interval', 5))
        self.spin_hybrid_update_unit = str(cfg(config, 'spin_hybrid_update_unit', 'epoch')).lower()
        self.spin_pop_buffer_size = int(cfg(config, 'spin_pop_buffer_size', 500))
        self.spin_niche_buffer_size = int(cfg(config, 'spin_niche_buffer_size', 500))
        self.spin_calib_candidate_size = int(cfg(config, 'spin_calib_candidate_size', 32))
        self.spin_calib_topk = int(cfg(config, 'spin_calib_topk', 3))
        self.spin_detach_calib_target = bool(cfg(config, 'spin_detach_calib_target', True))
        self.spin_calib_popularity_weight = float(cfg(config, 'spin_calib_popularity_weight', 0.1))
        self.spin_calib_niche_pop_penalty = float(cfg(config, 'spin_calib_niche_pop_penalty', 0.1))
        self.spin_niche_buffer_mode = str(cfg(config, 'spin_niche_buffer_mode', 'exclude_top_popular')).lower()
        self.spin_niche_exclude_top_ratio = float(cfg(config, 'spin_niche_exclude_top_ratio', 0.2))
        if self.spin_score_mode not in ('residual', 'two_stream'):
            raise ValueError(f'Unsupported spin_score_mode: {self.spin_score_mode}')
        if self.spin_hybrid_update_unit != 'epoch':
            raise ValueError(f'Unsupported spin_hybrid_update_unit: {self.spin_hybrid_update_unit}')

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
        self._build_positive_flow()
        self._build_anti_prototype_modules()
        self._build_hybrid_calibration_buffer()
        self.prototype_scorer = PrototypeScorer()
        self.current_epoch = -1

        self.last_user_pop = None
        self.last_user_niche = None
        self.last_item_pop = None
        self.last_item_niche = None
        self.last_user_route = None
        self.last_item_route = None
        self.last_user_rep = None
        self.last_user_out = None
        self.last_item_out = None
        self.last_z_pos_pop = None
        self.last_z_pos_niche = None
        self.last_z_neg_pop = None
        self.last_z_neg_niche = None
        self.last_pos_flow_loss = None
        self.last_pos_align_loss = None
        self.last_anti_gate_pop = None
        self.last_anti_gate_niche = None
        self.last_moe_gate_pop = None
        self.last_moe_gate_niche = None
        self.last_loss_rank = None
        self.last_loss_pos_flow = None
        self.last_loss_pos_align = None
        self.last_loss_anti_sep = None
        self.last_loss_anti_norm = None
        self.last_loss_anti_calib = None
        self.last_loss_aux = None
        self.last_loss_total = None
        self.last_loss_moe_entropy = None
        self.last_loss_moe_balance = None

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
        self.user_splitter = SoftPopularNicheSplitter(
            num_bands=self.num_freq_bands,
            embed_dim=self.all_dim,
            hidden_dim=self.spin_router_hidden_dim,
            dropout=self.spin_router_dropout,
        )
        self.item_splitter = SoftPopularNicheSplitter(
            num_bands=self.num_freq_bands,
            embed_dim=self.all_dim,
            hidden_dim=self.spin_router_hidden_dim,
            dropout=self.spin_router_dropout,
        )
        self.user_stream_gate = nn.Linear(self.all_dim, 2)
        self.item_stream_gate = nn.Linear(self.all_dim, 2)
        self.two_stream_scorer = TwoStreamScorer()

        nn.init.xavier_uniform_(self.user_stream_gate.weight)
        nn.init.zeros_(self.user_stream_gate.bias)
        nn.init.xavier_uniform_(self.item_stream_gate.weight)
        nn.init.zeros_(self.item_stream_gate.bias)

    def _resolved_velocity_type(self):
        if self.spin_flow_velocity_type == 'moe' and self.spin_moe_enable:
            return 'moe'
        return 'mlp'

    def _build_positive_flow(self):
        self.pos_condition_dim = self.all_dim * 3 + 1 + self.num_freq_bands
        self.pos_pop_flow = None
        self.pos_niche_flow = None
        self.pos_flow = None

        if not self.spin_enable_pos_flow:
            return

        flow_kwargs = dict(
            embed_dim=self.all_dim,
            condition_dim=self.pos_condition_dim,
            hidden_dim=self.spin_moe_hidden_dim,
            time_dim=self.spin_pos_flow_time_dim,
            velocity_type=self._resolved_velocity_type(),
            num_experts=self.spin_moe_num_experts,
            moe_router_hidden_dim=self.spin_moe_router_hidden_dim,
            moe_router_dropout=self.spin_moe_router_dropout,
            expert_dropout=self.spin_moe_expert_dropout,
            dense_moe=self.spin_moe_dense,
            normalize_output=self.spin_proto_normalize,
        )
        if self.spin_separate_pos_flow_heads:
            self.pos_pop_flow = PrototypeFlow(**flow_kwargs)
            self.pos_niche_flow = PrototypeFlow(**flow_kwargs)
        else:
            self.pos_flow = PrototypeFlow(**flow_kwargs)

    def _build_anti_prototype_modules(self):
        self.anti_pop_generator = None
        self.anti_niche_generator = None
        if not self.spin_enable_anti_proto:
            return

        anti_kwargs = dict(
            embed_dim=self.all_dim,
            condition_dim=self.pos_condition_dim,
            hidden_dim=self.spin_anti_gate_hidden_dim,
            gate_type=self.spin_anti_gate_type,
            normalize_output=self.spin_anti_proto_normalize,
        )
        self.anti_pop_generator = AntiPrototypeGenerator(**anti_kwargs)
        self.anti_niche_generator = AntiPrototypeGenerator(**anti_kwargs)

    def _build_hybrid_calibration_buffer(self):
        self.hybrid_calibration_buffer = None
        if not self.spin_enable_anti_proto or not self.spin_enable_hybrid_calibration:
            return
        self.hybrid_calibration_buffer = LightweightHybridCalibrationBuffer(
            item_popularity=self.item_popularity,
            pop_buffer_size=self.spin_pop_buffer_size,
            niche_buffer_size=self.spin_niche_buffer_size,
            calib_candidate_size=self.spin_calib_candidate_size,
            calib_topk=self.spin_calib_topk,
            calib_popularity_weight=self.spin_calib_popularity_weight,
            calib_niche_pop_penalty=self.spin_calib_niche_pop_penalty,
            niche_buffer_mode=self.spin_niche_buffer_mode,
            niche_exclude_top_ratio=self.spin_niche_exclude_top_ratio,
        )

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

    def _get_pos_flow_head(self, is_pop):
        if self.spin_separate_pos_flow_heads:
            return self.pos_pop_flow if is_pop else self.pos_niche_flow
        return self.pos_flow

    def _get_route_weights(self, route_info, route_key, indices, ref_tensor):
        if route_info is None:
            return ref_tensor.new_zeros(indices.size(0), self.num_freq_bands)
        route_weights = route_info.get(route_key)
        if route_weights is None:
            return ref_tensor.new_zeros(indices.size(0), self.num_freq_bands)
        return route_weights[indices]

    def _build_pos_condition(self, stream_repr, user_out, user_rep, user_activity, router_weights):
        return PrototypeFlow.build_condition(
            stream_repr=stream_repr,
            user_out=user_out,
            user_rep=user_rep,
            user_activity=user_activity,
            router_weights=router_weights,
            detach_condition=self.spin_detach_pos_condition,
        )

    def _build_source_seed(self, stream_repr, noise_std):
        x0 = stream_repr
        if noise_std > 0:
            x0 = x0 + noise_std * torch.randn_like(x0)
        if self.spin_proto_normalize:
            x0 = F.normalize(x0, dim=-1)
        return x0

    @staticmethod
    def _pair_score(left_emb, right_emb):
        return torch.sum(left_emb * right_emb, dim=-1)

    def _zero_scalar(self, ref_tensor):
        return ref_tensor.new_zeros(())

    def _reset_debug_cache(self):
        self.last_z_pos_pop = None
        self.last_z_pos_niche = None
        self.last_z_neg_pop = None
        self.last_z_neg_niche = None
        self.last_pos_flow_loss = None
        self.last_pos_align_loss = None
        self.last_anti_gate_pop = None
        self.last_anti_gate_niche = None
        self.last_moe_gate_pop = None
        self.last_moe_gate_niche = None
        self.last_loss_rank = None
        self.last_loss_pos_flow = None
        self.last_loss_pos_align = None
        self.last_loss_anti_sep = None
        self.last_loss_anti_norm = None
        self.last_loss_anti_calib = None
        self.last_loss_aux = None
        self.last_loss_total = None
        self.last_loss_moe_entropy = None
        self.last_loss_moe_balance = None

    def _cache_loss_debug(
        self,
        loss_rank,
        loss_pos_flow,
        loss_pos_align,
        loss_anti_sep,
        loss_anti_norm,
        loss_anti_calib,
        loss_aux,
        loss_total,
        loss_moe_entropy,
        loss_moe_balance,
    ):
        self.last_loss_rank = loss_rank.detach()
        self.last_loss_pos_flow = loss_pos_flow.detach()
        self.last_pos_flow_loss = loss_pos_flow.detach()
        self.last_loss_pos_align = loss_pos_align.detach()
        self.last_pos_align_loss = loss_pos_align.detach()
        self.last_loss_anti_sep = loss_anti_sep.detach()
        self.last_loss_anti_norm = loss_anti_norm.detach()
        self.last_loss_anti_calib = loss_anti_calib.detach()
        self.last_loss_aux = loss_aux.detach()
        self.last_loss_total = loss_total.detach()
        self.last_loss_moe_entropy = loss_moe_entropy.detach()
        self.last_loss_moe_balance = loss_moe_balance.detach()

    def _compute_base_pair_scores(
        self,
        user_out,
        pos_item_out,
        neg_item_out,
        user_pop,
        user_niche,
        pos_item_pop,
        pos_item_niche,
        neg_item_pop,
        neg_item_niche,
    ):
        if self.spin_score_mode == 'residual':
            pos_score = self.two_stream_scorer.score_pair(
                mode='residual',
                user_out=user_out,
                item_out=pos_item_out,
            )
            neg_score = self.two_stream_scorer.score_pair(
                mode='residual',
                user_out=user_out,
                item_out=neg_item_out,
            )
            return pos_score, neg_score

        pos_score = self.two_stream_scorer.score_pair(
            mode='two_stream',
            user_pop=user_pop,
            item_pop=pos_item_pop,
            user_niche=user_niche,
            item_niche=pos_item_niche,
            alpha=self.spin_pop_alpha,
            beta=self.spin_niche_beta,
        )
        neg_score = self.two_stream_scorer.score_pair(
            mode='two_stream',
            user_pop=user_pop,
            item_pop=neg_item_pop,
            user_niche=user_niche,
            item_niche=neg_item_niche,
            alpha=self.spin_pop_alpha,
            beta=self.spin_niche_beta,
        )
        return pos_score, neg_score

    def _compute_base_full_scores(self, user, user_out, item_out):
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

    def _compute_positive_prototypes(self, user_idx, target_pop=None, target_niche=None, noise_std=0.0):
        user_pop = self.last_user_pop[user_idx]
        user_niche = self.last_user_niche[user_idx]
        user_out = self.last_user_out[user_idx]
        user_rep = self.last_user_rep[user_idx]
        user_activity = self.user_activity[user_idx]
        pop_route = self._get_route_weights(self.last_user_route, 'pop_weights', user_idx, user_pop)
        niche_route = self._get_route_weights(self.last_user_route, 'niche_weights', user_idx, user_niche)

        x0_pop = self._build_source_seed(user_pop, noise_std)
        x0_niche = self._build_source_seed(user_niche, noise_std)
        cond_pop = self._build_pos_condition(user_pop, user_out, user_rep, user_activity, pop_route)
        cond_niche = self._build_pos_condition(user_niche, user_out, user_rep, user_activity, niche_route)

        flow_pop = self._get_pos_flow_head(is_pop=True)
        flow_niche = self._get_pos_flow_head(is_pop=False)
        zero = self._zero_scalar(user_out)

        if target_pop is not None and self.spin_detach_pos_target:
            target_pop = target_pop.detach()
        if target_niche is not None and self.spin_detach_pos_target:
            target_niche = target_niche.detach()

        if target_pop is None:
            pop_outputs = flow_pop.generate_one_step(x0_pop, cond_pop)
            pop_outputs.update({
                'flow_loss': zero,
                'v_pred': None,
                'v_target': None,
                'moe_entropy_loss': zero,
                'moe_balance_loss': zero,
            })
        else:
            pop_outputs = flow_pop(x0_pop, target_pop, cond_pop)

        if target_niche is None:
            niche_outputs = flow_niche.generate_one_step(x0_niche, cond_niche)
            niche_outputs.update({
                'flow_loss': zero,
                'v_pred': None,
                'v_target': None,
                'moe_entropy_loss': zero,
                'moe_balance_loss': zero,
            })
        else:
            niche_outputs = flow_niche(x0_niche, target_niche, cond_niche)

        z_pos_pop = pop_outputs['z_pos']
        z_pos_niche = niche_outputs['z_pos']

        align_loss_pop = zero if target_pop is None else prototype_cosine_alignment_loss(z_pos_pop, target_pop)
        align_loss_niche = zero if target_niche is None else prototype_cosine_alignment_loss(z_pos_niche, target_niche)

        return {
            'x0_pop': x0_pop,
            'x0_niche': x0_niche,
            'cond_pop': cond_pop,
            'cond_niche': cond_niche,
            'z_pos_pop': z_pos_pop,
            'z_pos_niche': z_pos_niche,
            'flow_loss': pop_outputs['flow_loss'] + niche_outputs['flow_loss'],
            'align_loss': align_loss_pop + align_loss_niche,
            'moe_entropy_loss': pop_outputs['moe_entropy_loss'] + niche_outputs['moe_entropy_loss'],
            'moe_balance_loss': pop_outputs['moe_balance_loss'] + niche_outputs['moe_balance_loss'],
            'moe_gate_pop': pop_outputs['moe_gate'],
            'moe_gate_niche': niche_outputs['moe_gate'],
        }

    def _compute_anti_prototypes(self, x0_pop, x0_niche, z_pos_pop, z_pos_niche, cond_pop, cond_niche):
        zero = self._zero_scalar(x0_pop)
        if not self.spin_enable_anti_proto:
            return {
                'z_neg_pop': None,
                'z_neg_niche': None,
                'gate_pop': None,
                'gate_niche': None,
                'loss_anti_sep': zero,
                'loss_anti_norm': zero,
            }

        z_neg_pop, gate_pop = self.anti_pop_generator(
            x0=x0_pop,
            z_pos=z_pos_pop,
            cond=cond_pop,
            anti_eta=self.spin_anti_eta_pop,
        )
        z_neg_niche, gate_niche = self.anti_niche_generator(
            x0=x0_niche,
            z_pos=z_pos_niche,
            cond=cond_niche,
            anti_eta=self.spin_anti_eta_niche,
        )
        loss_anti_sep = anti_separation_loss(z_neg_pop, z_pos_pop, self.spin_anti_sep_margin)
        loss_anti_sep = loss_anti_sep + anti_separation_loss(z_neg_niche, z_pos_niche, self.spin_anti_sep_margin)
        loss_anti_norm = anti_norm_loss(z_neg_pop, x0_pop) + anti_norm_loss(z_neg_niche, x0_niche)
        return {
            'z_neg_pop': z_neg_pop,
            'z_neg_niche': z_neg_niche,
            'gate_pop': gate_pop,
            'gate_niche': gate_niche,
            'loss_anti_sep': loss_anti_sep,
            'loss_anti_norm': loss_anti_norm,
        }

    def _compute_hybrid_calibration_loss(self, user_idx, pos_idx, user_out_batch, u_niche_batch, z_neg_pop, z_neg_niche):
        zero = self._zero_scalar(user_out_batch)
        if not self.spin_enable_anti_proto or not self.spin_enable_hybrid_calibration or self.hybrid_calibration_buffer is None:
            return zero
        if z_neg_pop is None or z_neg_niche is None:
            return zero

        target_calib_pop, target_calib_niche = self.hybrid_calibration_buffer.build_calibration_targets(
            user_idx=user_idx,
            pos_idx=pos_idx,
            user_out_batch=user_out_batch,
            u_niche_batch=u_niche_batch,
            i_pop_all=self.last_item_pop,
            i_niche_all=self.last_item_niche,
            item_out_all=self.last_item_out,
            item_popularity=self.item_popularity.view(-1),
        )
        if self.spin_detach_calib_target:
            target_calib_pop = target_calib_pop.detach()
            target_calib_niche = target_calib_niche.detach()
        return anti_calibration_loss(z_neg_pop, target_calib_pop) + anti_calibration_loss(z_neg_niche, target_calib_niche)

    def pre_epoch_processing(self):
        self.current_epoch += 1
        if self.hybrid_calibration_buffer is None:
            return
        if self.spin_hybrid_update_interval <= 0:
            return
        if self.current_epoch % self.spin_hybrid_update_interval == 0:
            self.hybrid_calibration_buffer.refresh_buffers()

    def forward(self):
        # id_rep: [num_user + num_item, D0]
        # v_rep: [num_user + num_item, D0]
        # t_rep: [num_user + num_item, D0]
        id_rep, v_rep, t_rep = self._encode_modalities()

        all_rep = torch.cat([id_rep, v_rep, t_rep], dim=1)  # [num_user + num_item, D]
        user_rep = all_rep[:self.num_user]  # [num_user, D]
        item_rep = all_rep[self.num_user:]  # [num_item, D]

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
        self.last_user_rep = user_rep
        self.last_user_out = user_out
        self.last_item_out = item_out

        return user_out, item_out, aux_loss

    @staticmethod
    def pairwise_bce_loss(user_emb, pos_item_emb, neg_item_emb):
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        return F.binary_cross_entropy_with_logits(logits, labels)

    @staticmethod
    def pairwise_bce_loss_from_scores(pos_scores, neg_scores):
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def calculate_loss(self, interaction):
        user_idx, pos_idx, neg_idx = interaction[0], interaction[1], interaction[2]
        user_emb_all, item_emb_all, aux_loss = self.forward()
        self._reset_debug_cache()

        user_emb = user_emb_all[user_idx]
        pos_item_emb = item_emb_all[pos_idx]
        neg_item_emb = item_emb_all[neg_idx]
        u_pop_batch = self.last_user_pop[user_idx]
        u_niche_batch = self.last_user_niche[user_idx]
        i_pop_pos = self.last_item_pop[pos_idx]
        i_niche_pos = self.last_item_niche[pos_idx]
        i_pop_neg = self.last_item_pop[neg_idx]
        i_niche_neg = self.last_item_niche[neg_idx]
        zero = self._zero_scalar(user_emb)

        base_pos_score, base_neg_score = self._compute_base_pair_scores(
            user_out=user_emb,
            pos_item_out=pos_item_emb,
            neg_item_out=neg_item_emb,
            user_pop=u_pop_batch,
            user_niche=u_niche_batch,
            pos_item_pop=i_pop_pos,
            pos_item_niche=i_niche_pos,
            neg_item_pop=i_pop_neg,
            neg_item_niche=i_niche_neg,
        )

        if not self.spin_enable_pos_flow:
            loss_rank = self.pairwise_bce_loss_from_scores(base_pos_score, base_neg_score)
            total_loss = loss_rank + self.ib_weight * aux_loss
            self._cache_loss_debug(
                loss_rank=loss_rank,
                loss_pos_flow=zero,
                loss_pos_align=zero,
                loss_anti_sep=zero,
                loss_anti_norm=zero,
                loss_anti_calib=zero,
                loss_aux=aux_loss,
                loss_total=total_loss,
                loss_moe_entropy=zero,
                loss_moe_balance=zero,
            )
            return total_loss

        proto_outputs = self._compute_positive_prototypes(
            user_idx=user_idx,
            target_pop=i_pop_pos,
            target_niche=i_niche_pos,
            noise_std=self.spin_pos_source_noise_std,
        )
        anti_outputs = self._compute_anti_prototypes(
            x0_pop=proto_outputs['x0_pop'],
            x0_niche=proto_outputs['x0_niche'],
            z_pos_pop=proto_outputs['z_pos_pop'],
            z_pos_niche=proto_outputs['z_pos_niche'],
            cond_pop=proto_outputs['cond_pop'],
            cond_niche=proto_outputs['cond_niche'],
        )
        pos_score, neg_score = self.prototype_scorer.score_pair(
            base_pos_score=base_pos_score,
            base_neg_score=base_neg_score,
            z_pos_pop=proto_outputs['z_pos_pop'],
            z_pos_niche=proto_outputs['z_pos_niche'],
            z_neg_pop=anti_outputs['z_neg_pop'],
            z_neg_niche=anti_outputs['z_neg_niche'],
            i_pop_pos=i_pop_pos,
            i_niche_pos=i_niche_pos,
            i_pop_neg=i_pop_neg,
            i_niche_neg=i_niche_neg,
            pos_score_weight=self.spin_pos_score_weight,
            anti_score_weight=self.spin_anti_score_weight,
            use_pos_score=self.spin_use_pos_score,
            use_anti_score=self.spin_use_anti_score,
        )

        loss_rank = self.pairwise_bce_loss_from_scores(pos_score, neg_score)
        loss_anti_calib = self._compute_hybrid_calibration_loss(
            user_idx=user_idx,
            pos_idx=pos_idx,
            user_out_batch=user_emb,
            u_niche_batch=u_niche_batch,
            z_neg_pop=anti_outputs['z_neg_pop'],
            z_neg_niche=anti_outputs['z_neg_niche'],
        )

        self.last_z_pos_pop = proto_outputs['z_pos_pop']
        self.last_z_pos_niche = proto_outputs['z_pos_niche']
        self.last_z_neg_pop = anti_outputs['z_neg_pop']
        self.last_z_neg_niche = anti_outputs['z_neg_niche']
        self.last_anti_gate_pop = anti_outputs['gate_pop']
        self.last_anti_gate_niche = anti_outputs['gate_niche']
        self.last_moe_gate_pop = proto_outputs['moe_gate_pop']
        self.last_moe_gate_niche = proto_outputs['moe_gate_niche']

        total_loss = loss_rank + self.ib_weight * aux_loss
        total_loss = total_loss + self.spin_pos_flow_weight * proto_outputs['flow_loss']
        total_loss = total_loss + self.spin_pos_align_weight * proto_outputs['align_loss']
        total_loss = total_loss + self.spin_anti_sep_weight * anti_outputs['loss_anti_sep']
        total_loss = total_loss + self.spin_anti_norm_weight * anti_outputs['loss_anti_norm']
        total_loss = total_loss + self.spin_anti_calib_weight * loss_anti_calib
        total_loss = total_loss + self.spin_moe_entropy_weight * proto_outputs['moe_entropy_loss']
        total_loss = total_loss + self.spin_moe_balance_weight * proto_outputs['moe_balance_loss']
        self._cache_loss_debug(
            loss_rank=loss_rank,
            loss_pos_flow=proto_outputs['flow_loss'],
            loss_pos_align=proto_outputs['align_loss'],
            loss_anti_sep=anti_outputs['loss_anti_sep'],
            loss_anti_norm=anti_outputs['loss_anti_norm'],
            loss_anti_calib=loss_anti_calib,
            loss_aux=aux_loss,
            loss_total=total_loss,
            loss_moe_entropy=proto_outputs['moe_entropy_loss'],
            loss_moe_balance=proto_outputs['moe_balance_loss'],
        )
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_out, item_out, aux_loss = self.forward()
        self._reset_debug_cache()
        base_scores = self._compute_base_full_scores(user, user_out, item_out)
        zero = self._zero_scalar(user_out)

        if not self.spin_enable_pos_flow:
            self._cache_loss_debug(
                loss_rank=zero,
                loss_pos_flow=zero,
                loss_pos_align=zero,
                loss_anti_sep=zero,
                loss_anti_norm=zero,
                loss_anti_calib=zero,
                loss_aux=aux_loss,
                loss_total=zero,
                loss_moe_entropy=zero,
                loss_moe_balance=zero,
            )
            return base_scores

        proto_outputs = self._compute_positive_prototypes(
            user_idx=user,
            target_pop=None,
            target_niche=None,
            noise_std=0.0,
        )
        anti_outputs = self._compute_anti_prototypes(
            x0_pop=proto_outputs['x0_pop'],
            x0_niche=proto_outputs['x0_niche'],
            z_pos_pop=proto_outputs['z_pos_pop'],
            z_pos_niche=proto_outputs['z_pos_niche'],
            cond_pop=proto_outputs['cond_pop'],
            cond_niche=proto_outputs['cond_niche'],
        )
        scores = self.prototype_scorer.score_full(
            base_scores=base_scores,
            z_pos_pop=proto_outputs['z_pos_pop'],
            z_pos_niche=proto_outputs['z_pos_niche'],
            z_neg_pop=anti_outputs['z_neg_pop'],
            z_neg_niche=anti_outputs['z_neg_niche'],
            i_pop_all=self.last_item_pop,
            i_niche_all=self.last_item_niche,
            pos_score_weight=self.spin_pos_score_weight,
            anti_score_weight=self.spin_anti_score_weight,
            use_pos_score=self.spin_use_pos_score,
            use_anti_score=self.spin_use_anti_score,
            pop_alpha=self.spin_pop_alpha,
            niche_beta=self.spin_niche_beta,
        )

        self.last_z_pos_pop = proto_outputs['z_pos_pop']
        self.last_z_pos_niche = proto_outputs['z_pos_niche']
        self.last_z_neg_pop = anti_outputs['z_neg_pop']
        self.last_z_neg_niche = anti_outputs['z_neg_niche']
        self.last_anti_gate_pop = anti_outputs['gate_pop']
        self.last_anti_gate_niche = anti_outputs['gate_niche']
        self.last_moe_gate_pop = proto_outputs['moe_gate_pop']
        self.last_moe_gate_niche = proto_outputs['moe_gate_niche']
        self._cache_loss_debug(
            loss_rank=zero,
            loss_pos_flow=zero,
            loss_pos_align=zero,
            loss_anti_sep=anti_outputs['loss_anti_sep'],
            loss_anti_norm=anti_outputs['loss_anti_norm'],
            loss_anti_calib=zero,
            loss_aux=aux_loss,
            loss_total=zero,
            loss_moe_entropy=zero,
            loss_moe_balance=zero,
        )
        return scores


if __name__ == '__main__':
    splitter = SoftPopularNicheSplitter(num_bands=3, embed_dim=12, hidden_dim=8, dropout=0.0)
    scorer = TwoStreamScorer()
    proto_scorer = PrototypeScorer()
    print(f'SoftPopularNicheSplitter import smoke test: {splitter.__class__.__name__}')
    print(f'TwoStreamScorer import smoke test: {scorer.__class__.__name__}')
    print(f'PrototypeScorer import smoke test: {proto_scorer.__class__.__name__}')
    print('SPIN module import smoke test passed. Full forward requires the project training entrypoint with a real RecDataset/TrainDataLoader.')
