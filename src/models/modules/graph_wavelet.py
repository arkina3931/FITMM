# coding: utf-8
import math
from typing import Iterable, List, Sequence

import torch


def _as_sparse_coo(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.layout == torch.sparse_coo:
        return tensor.coalesce()
    if getattr(tensor, "is_sparse", False):
        return tensor.coalesce()
    return tensor.to_sparse().coalesce()


def sparse_eye(size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    index = torch.arange(size, device=device, dtype=torch.long)
    indices = torch.stack([index, index], dim=0)
    values = torch.ones(size, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, (size, size), device=device, dtype=dtype).coalesce()


def build_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Build L = I - D^{-1/2} A D^{-1/2} from a sparse square adjacency."""
    adj = _as_sparse_coo(adj)
    if adj.size(0) != adj.size(1):
        raise ValueError(f"Expected a square adjacency, got shape {tuple(adj.shape)}")

    device = adj.device
    dtype = adj.dtype if adj.dtype.is_floating_point else torch.float32
    adj = adj.to(device=device, dtype=dtype).coalesce()

    # KNN graphs in this repo are directed; symmetrizing keeps the Laplacian
    # suitable for an undirected graph heat kernel without eigendecomposition.
    adj = (0.5 * (adj + adj.transpose(0, 1))).coalesce()
    indices = adj.indices()
    values = adj.values()
    row, col = indices[0], indices[1]

    degree = torch.sparse.sum(adj, dim=1).to_dense().to(dtype=dtype)
    inv_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)
    inv_sqrt_degree = torch.where(degree > 0, inv_sqrt_degree, torch.zeros_like(inv_sqrt_degree))
    norm_values = values * inv_sqrt_degree[row] * inv_sqrt_degree[col]

    eye = sparse_eye(adj.size(0), device=device, dtype=dtype)
    lap_indices = torch.cat([eye.indices(), indices], dim=1)
    lap_values = torch.cat([eye.values(), -norm_values], dim=0)
    return torch.sparse_coo_tensor(lap_indices, lap_values, adj.shape, device=device, dtype=dtype).coalesce()


def estimate_lambda_max(laplacian: torch.Tensor, n_iter: int = 0) -> float:
    """Return lambda_max for a normalized Laplacian; default is the safe value 2.0."""
    if n_iter <= 0:
        return 2.0

    laplacian = _as_sparse_coo(laplacian)
    x = torch.randn(laplacian.size(0), 1, device=laplacian.device, dtype=laplacian.dtype)
    x = x / (x.norm() + 1e-12)
    for _ in range(n_iter):
        x = torch.sparse.mm(laplacian, x)
        x = x / (x.norm() + 1e-12)
    lx = torch.sparse.mm(laplacian, x)
    value = torch.sum(x * lx).detach().clamp_min(1e-6)
    return float(value.item())


def rescale_laplacian(laplacian: torch.Tensor, lambda_max: float) -> torch.Tensor:
    """Rescale L to L_tilde = 2L / lambda_max - I."""
    if lambda_max <= 0:
        raise ValueError(f"lambda_max must be positive, got {lambda_max}")

    laplacian = _as_sparse_coo(laplacian)
    device = laplacian.device
    dtype = laplacian.dtype
    eye = sparse_eye(laplacian.size(0), device=device, dtype=dtype)
    indices = torch.cat([laplacian.indices(), eye.indices()], dim=1)
    values = torch.cat([laplacian.values() * (2.0 / float(lambda_max)), -eye.values()], dim=0)
    return torch.sparse_coo_tensor(indices, values, laplacian.shape, device=device, dtype=dtype).coalesce()


def chebyshev_basis(rescaled_laplacian: torch.Tensor, x: torch.Tensor, K: int) -> List[torch.Tensor]:
    """Return [T_0(X), ..., T_K(X)] using sparse Chebyshev recursion."""
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")

    rescaled_laplacian = _as_sparse_coo(rescaled_laplacian).to(device=x.device, dtype=x.dtype).coalesce()
    basis = [x]
    if K == 0:
        return basis

    t_prev = x
    t_cur = torch.sparse.mm(rescaled_laplacian, x)
    basis.append(t_cur)
    for _ in range(2, K + 1):
        t_next = 2.0 * torch.sparse.mm(rescaled_laplacian, t_cur) - t_prev
        basis.append(t_next)
        t_prev, t_cur = t_cur, t_next
    return basis


def graph_wavelet_filter(x, coeffs: torch.Tensor) -> torch.Tensor:
    """Apply fixed Chebyshev coefficients to a precomputed basis."""
    if isinstance(x, torch.Tensor):
        if x.dim() < 3:
            basis = [x]
        else:
            basis = [x[k] for k in range(x.size(0))]
    else:
        basis = list(x)

    coeffs = torch.as_tensor(coeffs, device=basis[0].device, dtype=basis[0].dtype).view(-1)
    if coeffs.numel() > len(basis):
        raise ValueError(f"Need at least {coeffs.numel()} basis tensors, got {len(basis)}")

    out = torch.zeros_like(basis[0])
    for coeff, term in zip(coeffs, basis):
        out = out + coeff * term
    return out


def heat_kernel_chebyshev_coefficients(
    scales: Sequence[float],
    K: int,
    lambda_max: float = 2.0,
    device=None,
    dtype=torch.float32,
    method: str = "numeric",
    num_nodes: int = None,
) -> torch.Tensor:
    """Precompute Chebyshev coefficients for exp(-s * lambda).

    The Chebyshev variable is x in [-1, 1], with
    lambda(x) = lambda_max / 2 * (x + 1).
    """
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")

    scales = [float(scale) for scale in scales]
    if method == "scipy":
        try:
            import numpy as np
            from scipy.special import iv

            coeffs = []
            for scale in scales:
                alpha = scale * float(lambda_max) / 2.0
                row = []
                for k in range(K + 1):
                    factor = 1.0 if k == 0 else 2.0
                    row.append(factor * math.exp(-alpha) * ((-1.0) ** k) * float(iv(k, alpha)))
                coeffs.append(row)
            return torch.as_tensor(np.asarray(coeffs), device=device, dtype=dtype)
        except Exception:
            method = "numeric"

    if method != "numeric":
        raise ValueError(f"Unknown coefficient method: {method}")

    n_nodes = int(num_nodes or max(128, 4 * (K + 1)))
    j = torch.arange(n_nodes, device=device, dtype=dtype)
    theta = math.pi * (j + 0.5) / float(n_nodes)
    x = torch.cos(theta)
    lamb = 0.5 * float(lambda_max) * (x + 1.0)

    coeff_rows = []
    for scale in scales:
        values = torch.exp(-float(scale) * lamb)
        row = []
        for k in range(K + 1):
            factor = 1.0 if k == 0 else 2.0
            coeff = factor * torch.sum(values * torch.cos(float(k) * theta)) / float(n_nodes)
            row.append(coeff)
        coeff_rows.append(torch.stack(row, dim=0))
    return torch.stack(coeff_rows, dim=0)


def heat_kernel_responses(
    rescaled_laplacian: torch.Tensor,
    x: torch.Tensor,
    coeffs_by_scale: torch.Tensor,
) -> List[torch.Tensor]:
    coeffs_by_scale = coeffs_by_scale.to(device=x.device, dtype=x.dtype)
    basis = chebyshev_basis(rescaled_laplacian, x, coeffs_by_scale.size(1) - 1)
    return [graph_wavelet_filter(basis, coeffs_by_scale[i]) for i in range(coeffs_by_scale.size(0))]


def three_band_decomposition(
    rescaled_laplacian: torch.Tensor,
    x: torch.Tensor,
    scales: Sequence[float],
    coeffs_by_scale: torch.Tensor,
) -> List[torch.Tensor]:
    responses = heat_kernel_responses(rescaled_laplacian, x, coeffs_by_scale)

    def find_scale(target: float) -> int:
        for idx, scale in enumerate(scales):
            if abs(float(scale) - target) < 1e-6:
                return idx
        raise ValueError(f"Required heat-kernel scale {target} is missing from {list(scales)}")

    h_1 = responses[find_scale(1.0)]
    h_2 = responses[find_scale(2.0)]
    low_band = h_2
    mid_band = h_1 - h_2
    high_band = x - h_1
    return [low_band, mid_band, high_band]


def _smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        device=device,
        dtype=torch.long,
    )
    values = torch.ones(indices.size(1), device=device)
    adj = torch.sparse_coo_tensor(indices, values, (4, 4), device=device).coalesce()
    lap = build_normalized_laplacian(adj)
    lambda_max = estimate_lambda_max(lap)
    lap_tilde = rescale_laplacian(lap, lambda_max)
    x = torch.randn(4, 5, device=device)
    coeffs = heat_kernel_chebyshev_coefficients([0.5, 1.0, 2.0], 5, lambda_max, device=device)
    basis = chebyshev_basis(lap_tilde, x, 5)
    response = graph_wavelet_filter(basis, coeffs[0])
    bands = three_band_decomposition(lap_tilde, x, [0.5, 1.0, 2.0], coeffs)
    assert len(basis) == 6
    assert response.shape == x.shape
    assert all(band.shape == x.shape for band in bands)
    assert torch.isfinite(torch.stack([band.norm() for band in bands])).all()
    print("graph_wavelet smoke test passed")


if __name__ == "__main__":
    _smoke_test()
