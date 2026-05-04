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

        if self.num_bands != 3:
            raise ValueError(f"Prism Stage 1 supports exactly 3 wavelet bands, got {self.num_bands}")
        for required_scale in (1.0, 2.0):
            if all(abs(scale - required_scale) >= 1e-6 for scale in self.wavelet_scales):
                raise ValueError(f"prism_wavelet_scales must include {required_scale}, got {self.wavelet_scales}")
        if self.popularity_prior_type not in {"log_degree", "degree"}:
            raise ValueError(f"Unknown prism_popularity_prior: {self.popularity_prior_type}")

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
        }

    def _cache_file(self, dataset_path: str) -> Optional[str]:
        if not self.use_wavelet_cache or not dataset_path or not os.path.isdir(dataset_path):
            return None
        name = f"prism_stage1_cache_k{self.knn_k}_u{self.num_user}_i{self.num_item}.pt"
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
        return cache.get("graph_data")

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
        }

    def _build_item_popularity_prior(self, item_degree: torch.Tensor) -> torch.Tensor:
        if self.popularity_prior_type == "degree":
            base = item_degree
        else:
            base = torch.log1p(item_degree)
        return _minmax_normalize(base).view(self.num_item, 1)

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
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return user_final, item_final

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

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()
        user_e = user_all_embeddings[users]
        pos_e = item_all_embeddings[pos_items]
        neg_e = item_all_embeddings[neg_items]

        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        reg_loss = self.reg_loss(
            self.user_id_embedding(users),
            self.item_id_embedding(pos_items),
            self.item_id_embedding(neg_items),
        )
        return mf_loss + self.reg_weight * reg_loss.squeeze()

    def predict(self, interaction):
        users = interaction[0]
        items = interaction[1]
        user_all_embeddings, item_all_embeddings = self.forward()
        return torch.sum(user_all_embeddings[users] * item_all_embeddings[items], dim=-1)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_all_embeddings, item_all_embeddings = self.forward()
        return user_all_embeddings[users] @ item_all_embeddings.t()

    def post_epoch_processing(self):
        if self.last_gate_mean is None:
            return None
        # Diagnostic reference ranges. These are soft health checks, not hard
        # stopping rules; compare trends across epochs and datasets.
        # gate_mean: user-level popular/niche gate in [0, 1]. Around 0.2-0.8
        #   means both streams are used; near 1.0 means popular dominates, near
        #   0.0 means niche dominates, and staying near either edge may indicate
        #   gate saturation or stream collapse.
        # pop_norm / niche_norm: mean L2 norm after stream normalization. They
        #   should be close to 1.0; values near 0 suggest an empty/collapsed
        #   stream, while NaN/Inf indicates numerical instability.
        # low/mid/high_energy: mean squared activation of each wavelet band.
        #   They must be finite and non-negative. There is no universal absolute
        #   range; all near 0 suggests representation collapse, a persistently
        #   dominant low band suggests oversmoothing, and a dominant high band
        #   often means the representation is noise-sensitive.
        return (
            "[Prism] "
            f"gate_mean={self.last_gate_mean:.6f}, "
            f"pop_norm={self.last_pop_norm:.6f}, "
            f"niche_norm={self.last_niche_norm:.6f}, "
            f"low_energy={self.last_low_energy:.6f}, "
            f"mid_energy={self.last_mid_energy:.6f}, "
            f"high_energy={self.last_high_energy:.6f}"
        )

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
        }
    )
    model = Prism(config, FakeTrainData()).to(config["device"])
    interaction = [
        torch.tensor([0, 1, 2], device=config["device"], dtype=torch.long),
        torch.tensor([0, 1, 2], device=config["device"], dtype=torch.long),
        torch.tensor([3, 4, 0], device=config["device"], dtype=torch.long),
    ]
    loss = model.calculate_loss(interaction)
    scores = model.full_sort_predict([interaction[0]])
    point_scores = model.predict([interaction[0], interaction[1]])
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert scores.shape == (3, 5) and torch.isfinite(scores).all()
    assert point_scores.shape == (3,) and torch.isfinite(point_scores).all()
    print("prism smoke test passed")


if __name__ == "__main__":
    _smoke_test()
