# coding: utf-8
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from losses.flow_matching import flow_matching_loss
from losses.prototype_align import prototype_align_loss
from losses.prototype_diversity import slot_diversity_loss, slot_offdiag_similarity
from models.niche_slots import NicheSlotRouter, NicheTargetBuilder
from models.prototype_flow import PositivePrototypeFlow
from models.modules.graph_wavelet import (
    build_normalized_laplacian,
    heat_kernel_chebyshev_coefficients,
    sparse_eye,
    three_band_decomposition,
    rescale_laplacian,
)


def _scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        if len(value) == 1:
            return value[0]
    return value


def _as_bool(value, default=False) -> bool:
    value = _scalar(value, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_float_list(value, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, str):
        parts = value.strip().strip("[]").split(",")
        return [float(part.strip()) for part in parts if part.strip()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    min_v = torch.min(x)
    max_v = torch.max(x)
    denom = max_v - min_v
    if float(denom.detach().cpu()) <= 1e-12:
        return torch.zeros_like(x)
    return (x - min_v) / (denom + 1e-12)


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Prism(GeneralRecommender):
    """Prism Stage 1: graph wavelet popular-niche frequency backbone."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.dataset = dataset
        self.num_user = self.n_users
        self.num_item = self.n_items

        self.embedding_dim = int(_scalar(config["embedding_size"], 64))
        self.feat_embed_dim = int(_scalar(config["feat_embed_dim"], self.embedding_dim))
        self.num_ui_layers = int(_scalar(config["num_layers"], 2))
        self.num_mm_layers = int(_scalar(config["n_mm_layers"], 1))
        self.knn_k = int(_scalar(config["knn_k"], 10))
        self.mm_image_weight = float(_scalar(config["mm_image_weight"], 0.1))
        self.reg_weight = float(_scalar(config["reg_weight"], 1e-4))

        self.enable_graph_wavelet = _as_bool(config["prism_enable_graph_wavelet"], True)
        self.num_bands = int(_scalar(config["prism_wavelet_num_bands"], 3))
        self.wavelet_scales = _as_float_list(config["prism_wavelet_scales"], [0.5, 1.0, 2.0])
        self.cheb_order = int(_scalar(config["prism_wavelet_cheb_order"], 5))
        self.lambda_max = float(_scalar(config["prism_wavelet_lambda_max"], 2.0))
        self.use_wavelet_cache = _as_bool(config["prism_wavelet_cache"], True)
        self.popularity_prior_type = str(_scalar(config["prism_popularity_prior"], "log_degree"))
        self.gate_hidden_dim = int(_scalar(config["prism_gate_hidden_dim"], self.embedding_dim))
        self.gate_dropout = float(_scalar(config["prism_gate_dropout"], 0.1))
        self.enable_positive_flow = _as_bool(config["prism_enable_positive_flow"], True)
        self.flow_hidden_dim = int(_scalar(config["prism_flow_hidden_dim"], 128))
        self.flow_time_dim = int(_scalar(config["prism_flow_time_dim"], 32))
        self.flow_num_layers = int(_scalar(config["prism_flow_num_layers"], 2))
        self.flow_dropout = float(_scalar(config["prism_flow_dropout"], 0.0))
        self.flow_match_weight = float(_scalar(config["prism_flow_match_weight"], 0.005))
        self.flow_align_start_epoch = int(_scalar(config["prism_flow_align_start_epoch"], 5))
        self.proto_align_weight = float(_scalar(config["prism_proto_align_weight"], 0.005))
        self.proto_score_weight = float(_scalar(config["prism_proto_score_weight"], 0.01))
        self.detach_positive_target = _as_bool(config["prism_detach_positive_target"], True)
        self.detach_flow_condition = _as_bool(config["prism_detach_flow_condition"], True)
        self.use_activity_cond = _as_bool(config["prism_use_activity_cond"], True)
        self.use_history_popularity_cond = _as_bool(config["prism_use_history_popularity_cond"], True)
        self.proto_normalize = _as_bool(config["prism_proto_normalize"], True)
        self.enable_multi_niche_slots = self.enable_positive_flow and _as_bool(
            config["prism_enable_multi_niche_slots"], True
        )
        self.niche_num_slots = int(_scalar(config["prism_niche_num_slots"], 4))
        self.niche_gate_hidden_dim = int(_scalar(config["prism_niche_gate_hidden_dim"], 128))
        self.niche_gate_dropout = float(_scalar(config["prism_niche_gate_dropout"], 0.0))
        self.niche_gate_activation = str(_scalar(config["prism_niche_gate_activation"], "softmax")).lower()
        self.niche_gate_temperature = float(_scalar(config["prism_niche_gate_temperature"], 0.5))
        self.slot_assign_temperature = float(_scalar(config["prism_slot_assign_temperature"], 0.2))
        self.slot_target_eps = float(_scalar(config["prism_slot_target_eps"], 1.0e-8))
        self.detach_slot_target = _as_bool(config["prism_detach_slot_target"], True)
        self.slot_flow_weight = float(_scalar(config["prism_slot_flow_weight"], self.flow_match_weight))
        self.slot_align_weight = float(_scalar(config["prism_slot_align_weight"], self.proto_align_weight))
        self.slot_div_weight = float(_scalar(config["prism_slot_div_weight"], 0.001))
        self.slot_div_margin = float(_scalar(config["prism_slot_div_margin"], 0.3))
        self.slot_target_max_history_len = max(1, int(_scalar(config["prism_slot_target_max_history_len"], 100)))
        self.eval_item_chunk_size = int(_scalar(config["prism_eval_item_chunk_size"], 4096))
        self.current_epoch = 0
        self.loss_log_names = ["rank", "reg", "splitter", "flow_match", "proto_align"]
        if self.enable_multi_niche_slots:
            self.loss_log_names.append("slot_div")
        self._eval_target_cache = None

        if self.num_bands != 3:
            raise ValueError(f"Prism Stage 1 supports exactly 3 wavelet bands, got {self.num_bands}")
        for required_scale in (1.0, 2.0):
            if all(abs(scale - required_scale) >= 1e-6 for scale in self.wavelet_scales):
                raise ValueError(f"prism_wavelet_scales must include {required_scale}, got {self.wavelet_scales}")
        if self.popularity_prior_type not in {"log_degree", "degree"}:
            raise ValueError(f"Unknown prism_popularity_prior: {self.popularity_prior_type}")
        if self.enable_multi_niche_slots and self.niche_num_slots < 1:
            raise ValueError(f"prism_niche_num_slots must be positive, got {self.niche_num_slots}")

        self.user_id_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_embedding = None
        self.text_embedding = None
        self.image_proj = None
        self.text_proj = None
        self.image_user_preference = None
        self.text_user_preference = None
        self._build_modal_embeddings()

        self.user_modal_norm = nn.LayerNorm(self.embedding_dim)
        self.item_modal_norm = nn.LayerNorm(self.embedding_dim)
        self.item_graph_norm = nn.LayerNorm(self.embedding_dim)
        self.user_pop_norm = nn.LayerNorm(self.embedding_dim)
        self.user_niche_norm = nn.LayerNorm(self.embedding_dim)
        self.user_final_norm = nn.LayerNorm(self.embedding_dim)
        self.item_final_norm = nn.LayerNorm(self.embedding_dim)

        gate_hidden = max(1, self.gate_hidden_dim)
        self.user_gate = nn.Sequential(
            nn.Linear(3 * self.embedding_dim + 1, gate_hidden),
            nn.LeakyReLU(),
            nn.Dropout(self.gate_dropout),
            nn.Linear(gate_hidden, 1),
        )
        self.user_gate.apply(_init_linear)
        self.flow_pop = None
        self.flow_niche = None
        if self.enable_positive_flow:
            flow_kwargs = dict(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.flow_hidden_dim,
                time_dim=self.flow_time_dim,
                num_layers=self.flow_num_layers,
                dropout=self.flow_dropout,
                use_activity_cond=self.use_activity_cond,
                use_history_popularity_cond=self.use_history_popularity_cond,
                normalize_output=self.proto_normalize,
                detach_condition=self.detach_flow_condition,
            )
            self.flow_pop = PositivePrototypeFlow(**flow_kwargs)
            self.flow_niche = PositivePrototypeFlow(**flow_kwargs)
        self.niche_slot_router = None
        self.niche_target_builder = None
        if self.enable_multi_niche_slots:
            self.niche_slot_router = NicheSlotRouter(
                embedding_dim=self.embedding_dim,
                num_slots=self.niche_num_slots,
                hidden_dim=self.niche_gate_hidden_dim,
                dropout=self.niche_gate_dropout,
                activation=self.niche_gate_activation,
                temperature=self.niche_gate_temperature,
            )
            self.niche_target_builder = NicheTargetBuilder(
                assign_temperature=self.slot_assign_temperature,
                eps=self.slot_target_eps,
                detach_target=self.detach_slot_target,
            )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        dataset_path = os.path.abspath(str(config["data_path"] or "") + str(config["dataset"] or ""))
        self._build_or_load_graph_cache(dataset_path)
        coeffs = heat_kernel_chebyshev_coefficients(
            scales=self.wavelet_scales,
            K=self.cheb_order,
            lambda_max=self.lambda_max,
            device=self.device,
            dtype=torch.float32,
            method="numeric",
        )
        self.register_buffer("wavelet_coeffs", coeffs)

        self.last_gate_mean = None
        self.last_pop_norm = None
        self.last_niche_norm = None
        self.last_low_energy = None
        self.last_mid_energy = None
        self.last_high_energy = None
        self.last_flow_match = None
        self.last_proto_align = None
        self.last_z_pos_pop_norm = None
        self.last_z_pos_niche_norm = None
        self.last_target_pos_pop_norm = None
        self.last_target_pos_niche_norm = None
        self.last_proto_score_mean = None
        self.last_niche_gate_entropy = None
        self.last_niche_gate_max = None
        self.last_avg_active_slots = None
        self.last_slot_flow_loss = None
        self.last_slot_align_loss = None
        self.last_slot_div_loss = None
        self.last_slot_sim_offdiag = None
        self.last_empty_slot_ratio = None
        self._log_startup()

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_modal_embeddings(self) -> None:
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = self._make_feature_projector(self.v_feat.size(1))
            self.image_user_preference = nn.Parameter(torch.empty(self.num_user, self.embedding_dim, device=self.device))
            nn.init.xavier_uniform_(self.image_user_preference)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = self._make_feature_projector(self.t_feat.size(1))
            self.text_user_preference = nn.Parameter(torch.empty(self.num_user, self.embedding_dim, device=self.device))
            nn.init.xavier_uniform_(self.text_user_preference)

    def _make_feature_projector(self, in_dim: int) -> nn.Module:
        hidden = max(self.embedding_dim, 4 * self.embedding_dim)
        projector = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, self.embedding_dim),
        ).to(self.device)
        projector.apply(_init_linear)
        return projector

    def _build_or_load_graph_cache(self, dataset_path: str) -> None:
        inter_mat = self.dataset.inter_matrix(form="coo").astype(np.float32)
        meta = self._cache_meta(inter_mat)
        cache = self._try_load_cache(dataset_path, meta)
        if cache is not None:
            self._register_graph_buffers(cache)
            self.graph_nnz = int(cache.get("graph_nnz", self.ui_norm_adj._nnz()))
            return

        graph_data = self._build_graph_data(inter_mat)
        graph_data["graph_nnz"] = int(
            graph_data["ui_norm_adj"]._nnz()
            + (0 if graph_data["image_item_adj"] is None else graph_data["image_item_adj"]._nnz())
            + (0 if graph_data["text_item_adj"] is None else graph_data["text_item_adj"]._nnz())
        )
        self._register_graph_buffers(graph_data)
        self.graph_nnz = int(graph_data["graph_nnz"])
        self._try_save_cache(dataset_path, meta, graph_data)

    def _cache_meta(self, inter_mat) -> Dict[str, object]:
        return {
            "num_user": int(self.num_user),
            "num_item": int(self.num_item),
            "inter_nnz": int(inter_mat.nnz),
            "knn_k": int(self.knn_k),
            "has_image": bool(self.v_feat is not None),
            "has_text": bool(self.t_feat is not None),
            "image_shape": None if self.v_feat is None else tuple(int(v) for v in self.v_feat.shape),
            "text_shape": None if self.t_feat is None else tuple(int(v) for v in self.t_feat.shape),
            "lambda_max": float(self.lambda_max),
            "cheb_order": int(self.cheb_order),
            "scales": tuple(round(float(s), 6) for s in self.wavelet_scales),
            "popularity_prior": self.popularity_prior_type,
            "slot_target_max_history_len": int(self.slot_target_max_history_len),
        }

    def _cache_file(self, dataset_path: str) -> Optional[str]:
        if not self.use_wavelet_cache or not dataset_path or not os.path.isdir(dataset_path):
            return None
        name = (
            f"prism_stage3_cache_k{self.knn_k}_u{self.num_user}_i{self.num_item}"
            f"_L{self.slot_target_max_history_len}.pt"
        )
        return os.path.join(dataset_path, name)

    def _try_load_cache(self, dataset_path: str, meta: Dict[str, object]):
        cache_file = self._cache_file(dataset_path)
        if cache_file is None or not os.path.isfile(cache_file):
            return None
        try:
            cache = torch.load(cache_file, map_location=self.device)
        except Exception:
            return None
        if cache.get("meta") != meta:
            return None
        graph_data = cache.get("graph_data")
        if not isinstance(graph_data, dict):
            return None
        required = {
            "user_history_adj",
            "history_crow_indices",
            "history_col_indices",
            "history_values",
            "hist_pad",
            "hist_mask",
            "user_history_popularity_all",
        }
        if not required.issubset(set(graph_data.keys())):
            return None
        return graph_data

    def _try_save_cache(self, dataset_path: str, meta: Dict[str, object], graph_data: Dict[str, object]) -> None:
        cache_file = self._cache_file(dataset_path)
        if cache_file is None:
            return

        def to_cpu(value):
            if torch.is_tensor(value):
                value = value.detach()
                if value.layout == torch.sparse_coo:
                    value = value.coalesce()
                return value.cpu()
            return value

        try:
            payload = {
                "meta": meta,
                "graph_data": {name: to_cpu(value) for name, value in graph_data.items()},
            }
            torch.save(payload, cache_file)
        except Exception:
            pass

    def _register_graph_buffers(self, graph_data: Dict[str, object]) -> None:
        for name in (
            "user_item_norm_adj",
            "ui_norm_adj",
            "image_item_adj",
            "text_item_adj",
            "item_mm_adj",
            "item_laplacian",
            "item_rescaled_laplacian",
            "item_popularity_prior",
            "user_activity",
            "user_history_adj",
            "history_crow_indices",
            "history_col_indices",
            "history_values",
            "hist_pad",
            "hist_mask",
            "user_history_popularity_all",
        ):
            value = graph_data[name]
            if torch.is_tensor(value):
                value = value.to(self.device)
                if value.layout == torch.sparse_coo:
                    value = value.coalesce()
            self.register_buffer(name, value)

    def _build_graph_data(self, inter_mat) -> Dict[str, object]:
        rows = torch.from_numpy(inter_mat.row.astype(np.int64)).to(self.device)
        cols = torch.from_numpy(inter_mat.col.astype(np.int64)).to(self.device)
        values = torch.ones(rows.numel(), device=self.device, dtype=torch.float32)

        user_degree = torch.bincount(rows, minlength=self.num_user).to(dtype=torch.float32, device=self.device)
        item_degree = torch.bincount(cols, minlength=self.num_item).to(dtype=torch.float32, device=self.device)
        item_popularity_prior = self._build_item_popularity_prior(item_degree)
        user_activity = _minmax_normalize(torch.log1p(user_degree)).view(self.num_user, 1)

        user_item_norm_adj = self._build_user_item_norm_adj(rows, cols, user_degree, values)
        user_history_adj, history_crow_indices, history_col_indices, history_values, hist_pad, hist_mask = (
            self._build_history_graph(inter_mat)
        )
        user_history_popularity_all = torch.sparse.mm(user_history_adj, item_popularity_prior)
        ui_norm_adj = self._build_square_ui_norm_adj(rows, cols, user_degree, item_degree, values)

        raw_item_graph_parts = []
        norm_item_graph_parts = []
        image_item_adj = None
        text_item_adj = None

        if self.image_embedding is not None:
            image_raw_adj = self._build_knn_raw_adj(self.image_embedding.weight.detach())
            image_item_adj = self._normalize_sparse_adj(image_raw_adj)
            image_weight = self.mm_image_weight if self.text_embedding is not None else 1.0
            raw_item_graph_parts.append((image_weight, image_raw_adj))
            norm_item_graph_parts.append((image_weight, image_item_adj))

        if self.text_embedding is not None:
            text_raw_adj = self._build_knn_raw_adj(self.text_embedding.weight.detach())
            text_item_adj = self._normalize_sparse_adj(text_raw_adj)
            text_weight = 1.0 if self.image_embedding is None else 1.0 - self.mm_image_weight
            raw_item_graph_parts.append((text_weight, text_raw_adj))
            norm_item_graph_parts.append((text_weight, text_item_adj))

        if raw_item_graph_parts:
            raw_item_adj = self._weighted_sparse_sum(raw_item_graph_parts, (self.num_item, self.num_item))
            item_mm_adj = self._weighted_sparse_sum(norm_item_graph_parts, (self.num_item, self.num_item))
        else:
            raw_item_adj = sparse_eye(self.num_item, device=self.device, dtype=torch.float32)
            item_mm_adj = sparse_eye(self.num_item, device=self.device, dtype=torch.float32)

        item_laplacian = build_normalized_laplacian(raw_item_adj)
        item_rescaled_laplacian = rescale_laplacian(item_laplacian, self.lambda_max)

        return {
            "user_item_norm_adj": user_item_norm_adj,
            "ui_norm_adj": ui_norm_adj,
            "image_item_adj": image_item_adj,
            "text_item_adj": text_item_adj,
            "item_mm_adj": item_mm_adj,
            "item_laplacian": item_laplacian,
            "item_rescaled_laplacian": item_rescaled_laplacian,
            "item_popularity_prior": item_popularity_prior,
            "user_activity": user_activity,
            "user_history_adj": user_history_adj,
            "history_crow_indices": history_crow_indices,
            "history_col_indices": history_col_indices,
            "history_values": history_values,
            "hist_pad": hist_pad,
            "hist_mask": hist_mask,
            "user_history_popularity_all": user_history_popularity_all,
        }

    def _build_item_popularity_prior(self, item_degree: torch.Tensor) -> torch.Tensor:
        if self.popularity_prior_type == "degree":
            base = item_degree
        else:
            base = torch.log1p(item_degree)
        return _minmax_normalize(base).view(self.num_item, 1)

    def _build_history_graph(self, inter_mat):
        csr = inter_mat.tocsr().astype(np.float32)
        row_counts = np.diff(csr.indptr).astype(np.float32)
        row_ids = np.repeat(np.arange(self.num_user, dtype=np.int64), np.diff(csr.indptr))
        if csr.data.size > 0:
            csr.data = np.ones_like(csr.data, dtype=np.float32)
            csr.data = csr.data / np.maximum(row_counts[row_ids], 1.0)

        crow = torch.from_numpy(csr.indptr.astype(np.int64)).to(self.device)
        col = torch.from_numpy(csr.indices.astype(np.int64)).to(self.device)
        val = torch.from_numpy(csr.data.astype(np.float32)).to(self.device)
        row = torch.from_numpy(row_ids).to(self.device)
        indices = torch.stack([row, col], dim=0) if row.numel() > 0 else torch.empty(2, 0, device=self.device, dtype=torch.long)
        history_adj = torch.sparse_coo_tensor(
            indices,
            val,
            (self.num_user, self.num_item),
            device=self.device,
            dtype=torch.float32,
        ).coalesce()
        max_len = self.slot_target_max_history_len
        hist_pad_np = np.zeros((self.num_user, max_len), dtype=np.int64)
        hist_mask_np = np.zeros((self.num_user, max_len), dtype=np.bool_)
        for user_idx in range(self.num_user):
            start = int(csr.indptr[user_idx])
            end = int(csr.indptr[user_idx + 1])
            user_items = csr.indices[start : min(end, start + max_len)]
            length = int(user_items.shape[0])
            if length > 0:
                hist_pad_np[user_idx, :length] = user_items
                hist_mask_np[user_idx, :length] = True

        hist_pad = torch.from_numpy(hist_pad_np).to(self.device)
        hist_mask = torch.from_numpy(hist_mask_np).to(self.device)
        return history_adj, crow, col, val, hist_pad, hist_mask

    def _build_user_item_norm_adj(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        user_degree: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if rows.numel() == 0:
            empty_indices = torch.empty(2, 0, device=self.device, dtype=torch.long)
            empty_values = torch.empty(0, device=self.device, dtype=torch.float32)
            return torch.sparse_coo_tensor(
                empty_indices, empty_values, (self.num_user, self.num_item), device=self.device
            ).coalesce()

        norm_values = values / user_degree[rows].clamp_min(1.0)
        indices = torch.stack([rows, cols], dim=0)
        return torch.sparse_coo_tensor(
            indices, norm_values, (self.num_user, self.num_item), device=self.device
        ).coalesce()

    def _build_square_ui_norm_adj(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        user_degree: torch.Tensor,
        item_degree: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if rows.numel() == 0:
            empty_indices = torch.empty(2, 0, device=self.device, dtype=torch.long)
            empty_values = torch.empty(0, device=self.device, dtype=torch.float32)
            return torch.sparse_coo_tensor(
                empty_indices,
                empty_values,
                (self.num_user + self.num_item, self.num_user + self.num_item),
                device=self.device,
            ).coalesce()

        norm_values = values / torch.sqrt(user_degree[rows].clamp_min(1.0) * item_degree[cols].clamp_min(1.0))
        user_to_item = torch.stack([rows, cols + self.num_user], dim=0)
        item_to_user = torch.stack([cols + self.num_user, rows], dim=0)
        indices = torch.cat([user_to_item, item_to_user], dim=1)
        norm_values = torch.cat([norm_values, norm_values], dim=0)
        return torch.sparse_coo_tensor(
            indices,
            norm_values,
            (self.num_user + self.num_item, self.num_user + self.num_item),
            device=self.device,
        ).coalesce()

    def _build_knn_raw_adj(self, embeddings: torch.Tensor) -> torch.Tensor:
        k = max(1, min(self.knn_k, embeddings.size(0)))
        normed = F.normalize(embeddings.to(self.device, dtype=torch.float32), p=2, dim=-1)
        sim = normed @ normed.t()
        _, knn_ind = torch.topk(sim, k, dim=-1)
        row = torch.arange(knn_ind.size(0), device=self.device).unsqueeze(1).expand_as(knn_ind)
        indices = torch.stack([row.reshape(-1), knn_ind.reshape(-1)], dim=0)
        values = torch.ones(indices.size(1), device=self.device, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (self.num_item, self.num_item), device=self.device).coalesce()

    def _normalize_sparse_adj(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.coalesce().to(self.device, dtype=torch.float32)
        indices = adj.indices()
        values = adj.values()
        row, col = indices[0], indices[1]
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(dtype=torch.float32)
        inv_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)
        inv_sqrt_degree = torch.where(degree > 0, inv_sqrt_degree, torch.zeros_like(inv_sqrt_degree))
        norm_values = values * inv_sqrt_degree[row] * inv_sqrt_degree[col]
        return torch.sparse_coo_tensor(indices, norm_values, adj.shape, device=self.device).coalesce()

    def _weighted_sparse_sum(self, parts: Sequence[Tuple[float, torch.Tensor]], shape: Tuple[int, int]) -> torch.Tensor:
        indices = []
        values = []
        for weight, tensor in parts:
            tensor = tensor.coalesce().to(self.device, dtype=torch.float32)
            indices.append(tensor.indices())
            values.append(tensor.values() * float(weight))
        if not indices:
            empty_indices = torch.empty(2, 0, device=self.device, dtype=torch.long)
            empty_values = torch.empty(0, device=self.device, dtype=torch.float32)
            return torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=self.device).coalesce()
        return torch.sparse_coo_tensor(
            torch.cat(indices, dim=1),
            torch.cat(values, dim=0),
            shape,
            device=self.device,
        ).coalesce()

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    def _ui_propagate(self, user_x: torch.Tensor, item_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_x = torch.cat([user_x, item_x], dim=0)
        outs = [all_x]
        h = all_x
        for _ in range(self.num_ui_layers):
            h = torch.sparse.mm(self.ui_norm_adj, h)
            outs.append(h)
        out = torch.stack(outs, dim=0).sum(dim=0)
        return out[: self.num_user], out[self.num_user :]

    def _item_graph_propagate(self, item_x: torch.Tensor) -> torch.Tensor:
        h = item_x
        for _ in range(self.num_mm_layers):
            h = torch.sparse.mm(self.item_mm_adj, h)
        return h

    def _encode_modalities(self) -> Tuple[torch.Tensor, torch.Tensor]:
        user_parts = []
        item_parts = []

        id_user, id_item = self._ui_propagate(self.user_id_embedding.weight, self.item_id_embedding.weight)
        user_parts.append(id_user)
        item_parts.append(id_item)

        if self.image_embedding is not None:
            image_item = self.image_proj(self.image_embedding.weight)
            image_user, image_item = self._ui_propagate(self.image_user_preference, image_item)
            user_parts.append(image_user)
            item_parts.append(image_item)

        if self.text_embedding is not None:
            text_item = self.text_proj(self.text_embedding.weight)
            text_user, text_item = self._ui_propagate(self.text_user_preference, text_item)
            user_parts.append(text_user)
            item_parts.append(text_item)

        user_rep = torch.stack(user_parts, dim=0).mean(dim=0)
        item_rep = torch.stack(item_parts, dim=0).mean(dim=0)
        return self.user_modal_norm(user_rep), self.item_modal_norm(item_rep)

    def _decompose_item_signal(self, item_signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_graph_wavelet:
            zeros = torch.zeros_like(item_signal)
            return item_signal, zeros, zeros
        low_band, mid_band, high_band = three_band_decomposition(
            self.item_rescaled_laplacian,
            item_signal,
            self.wavelet_scales,
            self.wavelet_coeffs,
        )
        return low_band, mid_band, high_band

    # ------------------------------------------------------------------
    # Forward / loss / inference
    # ------------------------------------------------------------------
    def set_epoch(self, epoch_idx: int) -> None:
        self.current_epoch = int(epoch_idx)

    def clear_eval_target_cache(self) -> None:
        self._eval_target_cache = None

    def _stage1_forward(self) -> Dict[str, torch.Tensor]:
        _, item_rep = self._encode_modalities()
        item_graph_rep = self._item_graph_propagate(item_rep)
        item_signal = self.item_graph_norm(item_rep + item_graph_rep)

        low_band, mid_band, high_band = self._decompose_item_signal(item_signal)
        rho = self.item_popularity_prior.to(item_signal.device, item_signal.dtype)
        e_pop = F.normalize(rho * low_band + (1.0 - rho) * mid_band, dim=-1)
        e_niche = F.normalize((1.0 - rho) * high_band + rho * mid_band, dim=-1)

        user_id = self.user_id_embedding.weight
        raw_u_pop = torch.sparse.mm(self.user_item_norm_adj, e_pop)
        raw_u_niche = torch.sparse.mm(self.user_item_norm_adj, e_niche)
        u_pop = self.user_pop_norm(user_id + raw_u_pop)
        u_niche = self.user_niche_norm(user_id + raw_u_niche)

        activity = self.user_activity.to(user_id.device, user_id.dtype)
        gate_input = torch.cat([user_id, u_pop, u_niche, activity], dim=-1)
        gate = torch.sigmoid(self.user_gate(gate_input))
        user_final = self.user_final_norm(user_id + gate * u_pop + (1.0 - gate) * u_niche)

        item_id = self.item_id_embedding.weight
        item_mix = 0.5 * e_pop + 0.5 * e_niche
        item_final = self.item_final_norm(item_id + item_mix)

        self._update_diagnostics(gate, e_pop, e_niche, low_band, mid_band, high_band)
        return {
            "user_final": user_final,
            "item_final": item_final,
            "user_popular": u_pop,
            "user_niche": u_niche,
            "item_popular": e_pop,
            "item_niche": e_niche,
            "user_base": user_id,
        }

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self._stage1_forward()
        return bundle["user_final"], bundle["item_final"]

    def _update_diagnostics(
        self,
        gate: torch.Tensor,
        e_pop: torch.Tensor,
        e_niche: torch.Tensor,
        low_band: torch.Tensor,
        mid_band: torch.Tensor,
        high_band: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.last_gate_mean = float(gate.mean().detach().cpu())
            self.last_pop_norm = float(e_pop.norm(dim=-1).mean().detach().cpu())
            self.last_niche_norm = float(e_niche.norm(dim=-1).mean().detach().cpu())
            self.last_low_energy = float(low_band.pow(2).mean().detach().cpu())
            self.last_mid_energy = float(mid_band.pow(2).mean().detach().cpu())
            self.last_high_energy = float(high_band.pow(2).mean().detach().cpu())

    def _build_user_history_adj_batch(self, users: torch.Tensor) -> torch.Tensor:
        users = users.to(self.device, dtype=torch.long)
        starts = self.history_crow_indices[users]
        ends = self.history_crow_indices[users + 1]
        counts = (ends - starts).to(dtype=torch.long)
        if int(counts.sum().detach().cpu()) == 0:
            empty_indices = torch.empty(2, 0, device=self.device, dtype=torch.long)
            empty_values = torch.empty(0, device=self.device, dtype=torch.float32)
            return torch.sparse_coo_tensor(
                empty_indices,
                empty_values,
                (users.numel(), self.num_item),
                device=self.device,
            ).coalesce()

        slices = []
        for start, end in zip(starts.detach().cpu().tolist(), ends.detach().cpu().tolist()):
            if end > start:
                slices.append(torch.arange(start, end, device=self.device, dtype=torch.long))
        positions = torch.cat(slices, dim=0)
        row = torch.repeat_interleave(torch.arange(users.numel(), device=self.device), counts)
        col = self.history_col_indices[positions]
        val = self.history_values[positions]
        return torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0),
            val,
            (users.numel(), self.num_item),
            device=self.device,
        ).coalesce()

    @staticmethod
    def _normalize_proto_target(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def _get_eval_target_cache(self, bundle: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._eval_target_cache is None:
            with torch.no_grad():
                target_pos_pop_all = self._normalize_proto_target(
                    torch.sparse.mm(self.user_history_adj, bundle["item_popular"].detach())
                )
                target_pos_niche_all = self._normalize_proto_target(
                    torch.sparse.mm(self.user_history_adj, bundle["item_niche"].detach())
                )
                self._eval_target_cache = {
                    "target_pos_pop_all": target_pos_pop_all.detach(),
                    "target_pos_niche_all": target_pos_niche_all.detach(),
                }
        return self._eval_target_cache

    def _get_positive_targets(
        self,
        users: torch.Tensor,
        bundle: Dict[str, torch.Tensor],
        use_eval_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        users = users.to(self.device, dtype=torch.long)
        if use_eval_cache:
            cache = self._get_eval_target_cache(bundle)
            target_pos_pop = cache["target_pos_pop_all"][users]
            target_pos_niche = cache["target_pos_niche_all"][users]
        else:
            history_adj_batch = self._build_user_history_adj_batch(users)
            target_pos_pop = self._normalize_proto_target(torch.sparse.mm(history_adj_batch, bundle["item_popular"]))
            target_pos_niche = self._normalize_proto_target(torch.sparse.mm(history_adj_batch, bundle["item_niche"]))

        if self.detach_positive_target or use_eval_cache:
            target_pos_pop = target_pos_pop.detach()
            target_pos_niche = target_pos_niche.detach()

        user_activity = self.user_activity[users].to(device=self.device, dtype=bundle["user_base"].dtype)
        user_history_popularity = self.user_history_popularity_all[users].to(
            device=self.device, dtype=bundle["user_base"].dtype
        )
        return target_pos_pop, target_pos_niche, user_activity, user_history_popularity

    def _generate_positive_prototypes(
        self,
        users: torch.Tensor,
        bundle: Dict[str, torch.Tensor],
        use_eval_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        target_pos_pop, target_pos_niche, user_activity, user_history_popularity = self._get_positive_targets(
            users, bundle, use_eval_cache=use_eval_cache
        )
        user_base = bundle["user_base"][users]
        user_popular = bundle["user_popular"][users]
        user_niche = bundle["user_niche"][users]
        z_pos_pop = self.flow_pop.generate(
            stream_user=user_popular,
            user_base=user_base,
            history_context=target_pos_pop,
            user_activity=user_activity,
            user_history_popularity=user_history_popularity,
        )

        proto_bundle = {
            "z_pos_pop": z_pos_pop,
            "target_pos_pop": target_pos_pop,
            "target_pos_niche": target_pos_niche,
            "user_activity": user_activity,
            "user_history_popularity": user_history_popularity,
            "user_base": user_base,
            "user_popular": user_popular,
            "user_niche": user_niche,
        }

        if self.enable_multi_niche_slots:
            slot_logits, niche_gate, user_niche_slots = self.niche_slot_router(
                user_base=user_base,
                user_niche=user_niche,
                user_activity=user_activity,
                user_history_popularity=user_history_popularity,
            )
            slot_embedding = self.niche_slot_router.slot_embedding_weight
            target_pos_niche_slots, slot_mass, empty_slot_ratio = self.niche_target_builder(
                users=users,
                item_niche=bundle["item_niche"],
                slot_embedding=slot_embedding,
                target_pos_niche=target_pos_niche,
                hist_pad=self.hist_pad,
                hist_mask=self.hist_mask,
            )
            z_pos_niche_slots = self.flow_niche.generate(
                stream_user=user_niche_slots,
                user_base=user_base,
                history_context=target_pos_niche_slots,
                user_activity=user_activity,
                user_history_popularity=user_history_popularity,
                slot_embedding=slot_embedding,
            )
            z_pos_niche_mix = (niche_gate.unsqueeze(-1) * z_pos_niche_slots).sum(dim=1)
            proto_bundle.update(
                {
                    "slot_logits": slot_logits,
                    "niche_gate": niche_gate,
                    "slot_embedding": slot_embedding,
                    "user_niche_slots": user_niche_slots,
                    "target_pos_niche_slots": target_pos_niche_slots,
                    "slot_mass": slot_mass,
                    "empty_slot_ratio": empty_slot_ratio,
                    "z_pos_niche_slots": z_pos_niche_slots,
                    "z_pos_niche_mix": z_pos_niche_mix,
                    "z_pos_niche": z_pos_niche_mix,
                }
            )
            self._update_stage3_generation_diagnostics(proto_bundle)
        else:
            z_pos_niche = self.flow_niche.generate(
                stream_user=user_niche,
                user_base=user_base,
                history_context=target_pos_niche,
                user_activity=user_activity,
                user_history_popularity=user_history_popularity,
            )
            proto_bundle.update({"z_pos_niche": z_pos_niche, "z_pos_niche_mix": z_pos_niche})

        self._update_flow_generation_diagnostics(
            proto_bundle["z_pos_pop"],
            proto_bundle["z_pos_niche_mix"],
            target_pos_pop,
            target_pos_niche,
        )
        return proto_bundle

    def _update_flow_generation_diagnostics(
        self,
        z_pos_pop: torch.Tensor,
        z_pos_niche: torch.Tensor,
        target_pos_pop: torch.Tensor,
        target_pos_niche: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.last_z_pos_pop_norm = float(z_pos_pop.norm(dim=-1).mean().detach().cpu())
            self.last_z_pos_niche_norm = float(z_pos_niche.norm(dim=-1).mean().detach().cpu())
            self.last_target_pos_pop_norm = float(target_pos_pop.norm(dim=-1).mean().detach().cpu())
            self.last_target_pos_niche_norm = float(target_pos_niche.norm(dim=-1).mean().detach().cpu())

    def _update_stage3_generation_diagnostics(self, proto_bundle: Dict[str, torch.Tensor]) -> None:
        niche_gate = proto_bundle["niche_gate"]
        z_pos_niche_slots = proto_bundle["z_pos_niche_slots"]
        with torch.no_grad():
            gate = niche_gate.detach()
            entropy = -(gate * gate.clamp_min(1e-12).log()).sum(dim=-1).mean()
            self.last_niche_gate_entropy = float(entropy.cpu())
            self.last_niche_gate_max = float(gate.max(dim=-1).values.mean().cpu())
            self.last_avg_active_slots = float((gate > 0.1).to(dtype=torch.float32).sum(dim=-1).mean().cpu())
            self.last_slot_sim_offdiag = float(slot_offdiag_similarity(z_pos_niche_slots).detach().cpu())
            self.last_empty_slot_ratio = float(proto_bundle["empty_slot_ratio"].detach().cpu())

    def _compute_flow_losses(
        self,
        users: torch.Tensor,
        bundle: Dict[str, torch.Tensor],
        proto_bundle: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        users = users.to(self.device, dtype=torch.long)
        user_base = proto_bundle["user_base"]
        user_activity = proto_bundle["user_activity"]
        user_history_popularity = proto_bundle["user_history_popularity"]

        x0_pop = proto_bundle["user_popular"].detach()
        x1_pop = proto_bundle["target_pos_pop"].detach()

        t_pop = torch.rand(users.numel(), 1, device=self.device, dtype=x0_pop.dtype)
        zt_pop = (1.0 - t_pop) * x0_pop + t_pop * x1_pop
        v_target_pop = x1_pop - x0_pop
        v_pred_pop, _ = self.flow_pop(
            stream_user=zt_pop,
            t=t_pop,
            user_base=user_base,
            history_context=x1_pop,
            user_activity=user_activity,
            user_history_popularity=user_history_popularity,
        )

        flow_pop = flow_matching_loss(v_pred_pop, v_target_pop)
        align_pop = prototype_align_loss(proto_bundle["z_pos_pop"], proto_bundle["target_pos_pop"].detach())

        if self.enable_multi_niche_slots:
            x0_niche = proto_bundle["user_niche_slots"].detach()
            x1_niche = proto_bundle["target_pos_niche_slots"].detach()
            batch_size, num_slots, dim = x0_niche.shape
            t_niche = torch.rand(batch_size, num_slots, 1, device=self.device, dtype=x0_niche.dtype)
            zt_niche = (1.0 - t_niche) * x0_niche + t_niche * x1_niche
            v_target_niche = x1_niche - x0_niche
            v_pred_niche, _ = self.flow_niche(
                stream_user=zt_niche,
                t=t_niche,
                user_base=user_base,
                history_context=x1_niche,
                user_activity=user_activity,
                user_history_popularity=user_history_popularity,
                slot_embedding=proto_bundle["slot_embedding"],
            )
            slot_flow_raw = flow_matching_loss(
                v_pred_niche.reshape(batch_size * num_slots, dim),
                v_target_niche.reshape(batch_size * num_slots, dim),
                reduction="none",
            ).view(batch_size, num_slots)
            flow_niche = (proto_bundle["niche_gate"].detach() * slot_flow_raw).sum(dim=-1).mean()

            align_slot_raw = 1.0 - F.cosine_similarity(
                proto_bundle["z_pos_niche_slots"],
                proto_bundle["target_pos_niche_slots"].detach(),
                dim=-1,
            )
            align_niche = (proto_bundle["niche_gate"].detach() * align_slot_raw).sum(dim=-1).mean()
            slot_div = slot_diversity_loss(
                proto_bundle["z_pos_niche_slots"],
                margin=self.slot_div_margin,
            )
        else:
            x0_niche = proto_bundle["user_niche"].detach()
            x1_niche = proto_bundle["target_pos_niche"].detach()
            t_niche = torch.rand(users.numel(), 1, device=self.device, dtype=x0_niche.dtype)
            zt_niche = (1.0 - t_niche) * x0_niche + t_niche * x1_niche
            v_target_niche = x1_niche - x0_niche
            v_pred_niche, _ = self.flow_niche(
                stream_user=zt_niche,
                t=t_niche,
                user_base=user_base,
                history_context=x1_niche,
                user_activity=user_activity,
                user_history_popularity=user_history_popularity,
            )
            flow_niche = flow_matching_loss(v_pred_niche, v_target_niche)
            align_niche = prototype_align_loss(
                proto_bundle["z_pos_niche"],
                proto_bundle["target_pos_niche"].detach(),
            )
            slot_div = flow_niche.new_zeros(())

        return flow_pop, flow_niche, align_pop, align_niche, slot_div

    def _score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        bundle: Dict[str, torch.Tensor],
        use_eval_cache: bool = False,
    ) -> torch.Tensor:
        base_score = torch.sum(bundle["user_final"][users] * bundle["item_final"][items], dim=-1)
        if not self.enable_positive_flow:
            return base_score
        proto_bundle = self._generate_positive_prototypes(users, bundle, use_eval_cache=use_eval_cache)
        proto_pop_score = torch.sum(proto_bundle["z_pos_pop"] * bundle["item_popular"][items], dim=-1)
        proto_niche_score = torch.sum(proto_bundle["z_pos_niche_mix"] * bundle["item_niche"][items], dim=-1)
        proto_score = proto_pop_score + proto_niche_score
        with torch.no_grad():
            self.last_proto_score_mean = float(proto_score.mean().detach().cpu())
        return base_score + self.proto_score_weight * proto_score

    def _score_all_items_chunked(
        self,
        users: torch.Tensor,
        bundle: Dict[str, torch.Tensor],
        proto_bundle: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        user_final = bundle["user_final"][users]
        scores = torch.empty(user_final.size(0), self.num_item, device=self.device, dtype=user_final.dtype)
        chunk_size = self.eval_item_chunk_size if self.eval_item_chunk_size > 0 else self.num_item
        proto_sum = user_final.new_zeros(())
        proto_count = 0
        for start in range(0, self.num_item, chunk_size):
            end = min(start + chunk_size, self.num_item)
            chunk_scores = user_final @ bundle["item_final"][start:end].t()
            if proto_bundle is not None:
                proto_chunk = (
                    proto_bundle["z_pos_pop"] @ bundle["item_popular"][start:end].t()
                    + proto_bundle["z_pos_niche_mix"] @ bundle["item_niche"][start:end].t()
                )
                chunk_scores = chunk_scores + self.proto_score_weight * proto_chunk
                with torch.no_grad():
                    proto_sum = proto_sum + proto_chunk.detach().sum()
                    proto_count += proto_chunk.numel()
            scores[:, start:end] = chunk_scores
        if proto_bundle is not None and proto_count > 0:
            with torch.no_grad():
                self.last_proto_score_mean = float((proto_sum / proto_count).detach().cpu())
        return scores

    def _score_all_items(
        self,
        users: torch.Tensor,
        bundle: Dict[str, torch.Tensor],
        use_eval_cache: bool = False,
    ) -> torch.Tensor:
        use_chunk = self.eval_item_chunk_size > 0 and self.num_item > self.eval_item_chunk_size
        if not self.enable_positive_flow:
            if use_chunk:
                return self._score_all_items_chunked(users, bundle, proto_bundle=None)
            return bundle["user_final"][users] @ bundle["item_final"].t()

        proto_bundle = self._generate_positive_prototypes(users, bundle, use_eval_cache=use_eval_cache)
        if use_chunk:
            return self._score_all_items_chunked(users, bundle, proto_bundle=proto_bundle)

        base_scores = bundle["user_final"][users] @ bundle["item_final"].t()
        proto_scores = proto_bundle["z_pos_pop"] @ bundle["item_popular"].t()
        proto_scores = proto_scores + proto_bundle["z_pos_niche_mix"] @ bundle["item_niche"].t()
        with torch.no_grad():
            self.last_proto_score_mean = float(proto_scores.mean().detach().cpu())
        return base_scores + self.proto_score_weight * proto_scores

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        bundle = self._stage1_forward()
        base_pos_scores = torch.sum(bundle["user_final"][users] * bundle["item_final"][pos_items], dim=1)
        base_neg_scores = torch.sum(bundle["user_final"][users] * bundle["item_final"][neg_items], dim=1)

        flow_pop = base_pos_scores.new_zeros(())
        flow_niche = base_pos_scores.new_zeros(())
        align_pop = base_pos_scores.new_zeros(())
        align_niche = base_pos_scores.new_zeros(())
        slot_div = base_pos_scores.new_zeros(())
        pos_scores = base_pos_scores
        neg_scores = base_neg_scores

        if self.enable_positive_flow:
            proto_bundle = self._generate_positive_prototypes(users, bundle, use_eval_cache=False)
            proto_pos_scores = (
                torch.sum(proto_bundle["z_pos_pop"] * bundle["item_popular"][pos_items], dim=1)
                + torch.sum(proto_bundle["z_pos_niche_mix"] * bundle["item_niche"][pos_items], dim=1)
            )
            proto_neg_scores = (
                torch.sum(proto_bundle["z_pos_pop"] * bundle["item_popular"][neg_items], dim=1)
                + torch.sum(proto_bundle["z_pos_niche_mix"] * bundle["item_niche"][neg_items], dim=1)
            )
            with torch.no_grad():
                self.last_proto_score_mean = float(
                    torch.cat([proto_pos_scores, proto_neg_scores], dim=0).mean().detach().cpu()
            )
            pos_scores = base_pos_scores + self.proto_score_weight * proto_pos_scores
            neg_scores = base_neg_scores + self.proto_score_weight * proto_neg_scores
            flow_pop, flow_niche, align_pop, align_niche, slot_div = self._compute_flow_losses(
                users,
                bundle,
                proto_bundle,
            )

        rank_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_id_embedding(users),
            self.item_id_embedding(pos_items),
            self.item_id_embedding(neg_items),
        ).squeeze()
        splitter_reg_loss = rank_loss.new_zeros(())
        flow_niche_weight = self.slot_flow_weight if self.enable_multi_niche_slots else self.flow_match_weight
        align_niche_weight = self.slot_align_weight if self.enable_multi_niche_slots else self.proto_align_weight
        flow_match = flow_pop + flow_niche
        proto_align = align_pop + align_niche
        flow_match_loss_weighted = self.flow_match_weight * flow_pop + flow_niche_weight * flow_niche
        if self.current_epoch >= self.flow_align_start_epoch:
            proto_align_loss_weighted = self.proto_align_weight * align_pop + align_niche_weight * align_niche
        else:
            proto_align_loss_weighted = proto_align * 0.0
        slot_div_loss_weighted = self.slot_div_weight * slot_div if self.enable_multi_niche_slots else slot_div * 0.0

        with torch.no_grad():
            self.last_flow_match = float(flow_match.detach().cpu())
            self.last_proto_align = float(proto_align.detach().cpu())
            if self.enable_multi_niche_slots:
                self.last_slot_flow_loss = float(flow_niche.detach().cpu())
                self.last_slot_align_loss = float(align_niche.detach().cpu())
                self.last_slot_div_loss = float(slot_div.detach().cpu())
        losses = (rank_loss, reg_loss, splitter_reg_loss, flow_match_loss_weighted, proto_align_loss_weighted)
        if self.enable_multi_niche_slots:
            losses = losses + (slot_div_loss_weighted,)
        return losses

    def predict(self, interaction):
        users = interaction[0]
        items = interaction[1]
        bundle = self._stage1_forward()
        return self._score_pairs(users, items, bundle, use_eval_cache=not self.training)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        bundle = self._stage1_forward()
        return self._score_all_items(users, bundle, use_eval_cache=not self.training)

    # def post_epoch_processing(self):
        # if self.last_gate_mean is None:
        #     return None
        # # Diagnostic reference ranges. These are soft health checks, not hard
        # # stopping rules; compare trends across epochs and datasets.
        # # gate_mean: user-level popular/niche gate in [0, 1]. Around 0.2-0.8
        # #   means both streams are used; near 1.0 means popular dominates, near
        # #   0.0 means niche dominates, and staying near either edge may indicate
        # #   gate saturation or stream collapse.
        # # pop_norm / niche_norm: mean L2 norm after stream normalization. They
        # #   should be close to 1.0; values near 0 suggest an empty/collapsed
        # #   stream, while NaN/Inf indicates numerical instability.
        # # low/mid/high_energy: mean squared activation of each wavelet band.
        # #   They must be finite and non-negative. There is no universal absolute
        # #   range; all near 0 suggests representation collapse, a persistently
        # #   dominant low band suggests oversmoothing, and a dominant high band
        # #   often means the representation is noise-sensitive.
        # stage2_info = ""
        # if self.last_flow_match is not None:
        #     stage2_info = (
        #         f", flow_match={self.last_flow_match:.6f}, "
        #         f"proto_align={self.last_proto_align:.6f}, "
        #         f"z_pos_pop_norm={self.last_z_pos_pop_norm:.6f}, "
        #         f"z_pos_niche_norm={self.last_z_pos_niche_norm:.6f}, "
        #         f"target_pos_pop_norm={self.last_target_pos_pop_norm:.6f}, "
        #         f"target_pos_niche_norm={self.last_target_pos_niche_norm:.6f}, "
        #         f"proto_score_mean={self.last_proto_score_mean:.6f}"
        #     )
        # stage3_info = ""
        # if self.last_slot_flow_loss is not None:
        #     stage3_info = (
        #         f", niche_gate_entropy={self.last_niche_gate_entropy:.6f}, "
        #         f"niche_gate_max={self.last_niche_gate_max:.6f}, "
        #         f"avg_active_slots={self.last_avg_active_slots:.6f}, "
        #         f"slot_flow_loss={self.last_slot_flow_loss:.6f}, "
        #         f"slot_align_loss={self.last_slot_align_loss:.6f}, "
        #         f"slot_div_loss={self.last_slot_div_loss:.6f}, "
        #         f"slot_sim_offdiag={self.last_slot_sim_offdiag:.6f}, "
        #         f"empty_slot_ratio={self.last_empty_slot_ratio:.6f}"
        #     )
        # return (
        #     "[Prism] "
        #     f"gate_mean={self.last_gate_mean:.6f}, "
        #     f"pop_norm={self.last_pop_norm:.6f}, "
        #     f"niche_norm={self.last_niche_norm:.6f}, "
        #     f"low_energy={self.last_low_energy:.6f}, "
        #     f"mid_energy={self.last_mid_energy:.6f}, "
        #     f"high_energy={self.last_high_energy:.6f}"
        #     f"{stage2_info}"
        #     f"{stage3_info}"
        # )

    def _log_startup(self) -> None:
        if self.enable_graph_wavelet:
            print("[Prism] Stage 1 graph wavelet backbone enabled")
        print(
            "[Prism] "
            f"num_bands={self.num_bands}, "
            f"scales={self.wavelet_scales}, "
            f"cheb_order={self.cheb_order}, "
            f"lambda_max={self.lambda_max}, "
            f"graph nnz={self.graph_nnz}, "
            f"popularity prior type={self.popularity_prior_type}"
        )
        if self.enable_positive_flow:
            print(
                "[Prism] Stage 2 positive prototype rectified flow enabled "
                f"flow_hidden_dim={self.flow_hidden_dim}, "
                f"time_dim={self.flow_time_dim}, "
                f"flow_match_weight={self.flow_match_weight}, "
                f"proto_align_weight={self.proto_align_weight}, "
                f"proto_score_weight={self.proto_score_weight}, "
                f"detach_positive_target={self.detach_positive_target}"
            )
        if self.enable_multi_niche_slots:
            print("[Prism] Stage 3 multi-slot niche prototype enabled")
            print(
                "[Prism] "
                f"niche_num_slots={self.niche_num_slots}, "
                f"niche_gate_activation={self.niche_gate_activation}, "
                f"niche_gate_temperature={self.niche_gate_temperature}, "
                f"slot_assign_temperature={self.slot_assign_temperature}, "
                f"slot_div_weight={self.slot_div_weight}, "
                f"slot_div_margin={self.slot_div_margin}, "
                f"slot_target_max_history_len={self.slot_target_max_history_len}, "
                f"eval_item_chunk_size={self.eval_item_chunk_size}"
            )


PRISM = Prism


def _smoke_test():
    from scipy.sparse import coo_matrix

    class FakeConfig(dict):
        def __getitem__(self, key):
            return self.get(key)

    class FakeDataset:
        def __init__(self):
            self.user_num = 4
            self.item_num = 5

        def get_user_num(self):
            return self.user_num

        def get_item_num(self):
            return self.item_num

    class FakeTrainData:
        def __init__(self):
            self.dataset = FakeDataset()

        def inter_matrix(self, form="coo", value_field=None):
            rows = np.array([0, 0, 1, 2, 2, 3], dtype=np.int64)
            cols = np.array([0, 1, 1, 2, 3, 4], dtype=np.int64)
            data = np.ones_like(rows, dtype=np.float32)
            mat = coo_matrix((data, (rows, cols)), shape=(4, 5))
            return mat if form == "coo" else mat.tocsr()

    config = FakeConfig(
        {
            "USER_ID_FIELD": "userID",
            "ITEM_ID_FIELD": "itemID",
            "NEG_PREFIX": "neg_",
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "train_batch_size": 4,
            "embedding_size": 8,
            "feat_embed_dim": 8,
            "num_layers": 1,
            "n_mm_layers": 1,
            "knn_k": 2,
            "mm_image_weight": 0.1,
            "reg_weight": 1e-4,
            "end2end": True,
            "is_multimodal_model": False,
            "data_path": "",
            "dataset": "fake",
            "prism_enable_graph_wavelet": True,
            "prism_wavelet_num_bands": 3,
            "prism_wavelet_scales": [0.5, 1.0, 2.0],
            "prism_wavelet_cheb_order": 3,
            "prism_wavelet_lambda_max": 2.0,
            "prism_wavelet_cache": False,
            "prism_popularity_prior": "log_degree",
            "prism_gate_hidden_dim": 8,
            "prism_gate_dropout": 0.0,
            "prism_enable_positive_flow": True,
            "prism_flow_hidden_dim": 16,
            "prism_flow_time_dim": 8,
            "prism_flow_num_layers": 2,
            "prism_flow_dropout": 0.0,
            "prism_flow_match_weight": 0.005,
            "prism_flow_align_start_epoch": 1,
            "prism_proto_align_weight": 0.005,
            "prism_proto_score_weight": 0.01,
            "prism_detach_positive_target": True,
            "prism_detach_flow_condition": True,
            "prism_use_activity_cond": True,
            "prism_use_history_popularity_cond": True,
            "prism_proto_normalize": True,
            "prism_enable_multi_niche_slots": True,
            "prism_niche_num_slots": 4,
            "prism_niche_gate_hidden_dim": 16,
            "prism_niche_gate_dropout": 0.0,
            "prism_niche_gate_activation": "softmax",
            "prism_niche_gate_temperature": 0.5,
            "prism_slot_assign_temperature": 0.2,
            "prism_slot_target_eps": 1.0e-8,
            "prism_detach_slot_target": True,
            "prism_slot_flow_weight": 0.005,
            "prism_slot_align_weight": 0.005,
            "prism_slot_div_weight": 0.001,
            "prism_slot_div_margin": 0.3,
            "prism_slot_target_max_history_len": 3,
            "prism_eval_item_chunk_size": 2,
        }
    )
    model = Prism(config, FakeTrainData()).to(config["device"])
    model.set_epoch(1)
    interaction = [
        torch.tensor([0, 1, 2], device=config["device"], dtype=torch.long),
        torch.tensor([0, 1, 2], device=config["device"], dtype=torch.long),
        torch.tensor([3, 4, 0], device=config["device"], dtype=torch.long),
    ]
    losses = model.calculate_loss(interaction)
    scores = model.full_sort_predict([interaction[0]])
    point_scores = model.predict([interaction[0], interaction[1]])
    history_batch = model._build_user_history_adj_batch(interaction[0])
    bundle = model._stage1_forward()
    proto_bundle = model._generate_positive_prototypes(interaction[0], bundle, use_eval_cache=False)
    assert len(losses) == 6 and all(loss.dim() == 0 and torch.isfinite(loss) for loss in losses)
    assert model.user_history_adj.layout == torch.sparse_coo
    assert history_batch.layout == torch.sparse_coo
    assert history_batch.shape == (3, 5)
    assert model.hist_pad.shape == (4, 3)
    assert model.hist_mask.shape == (4, 3)
    assert proto_bundle["z_pos_niche_slots"].shape == (3, 4, 8)
    assert torch.allclose(proto_bundle["niche_gate"].sum(dim=-1), torch.ones(3, device=config["device"]), atol=1e-5)
    assert proto_bundle["target_pos_niche_slots"].shape == (3, 4, 8)
    assert scores.shape == (3, 5) and torch.isfinite(scores).all()
    assert point_scores.shape == (3,) and torch.isfinite(point_scores).all()
    model.eval()
    _ = model.full_sort_predict([interaction[0]])
    assert model._eval_target_cache is not None
    model.clear_eval_target_cache()
    assert model._eval_target_cache is None
    print("prism smoke test passed")


if __name__ == "__main__":
    _smoke_test()
