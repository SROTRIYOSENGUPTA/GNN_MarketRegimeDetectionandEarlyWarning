"""
THGNN Dataset & DataLoader
===========================
Constructs daily stock-graph snapshots for the THGNN pipeline.

Key tensor shapes (per sample = one trading day t):
────────────────────────────────────────────────────
  node_features   : (N_t, L, F)       = (≤500, 30, 37)
      N_t  = number of eligible stocks on day t  (≤ 500)
      L    = 30-day lookback sequence length
      F    = 37 features

  edge_index      : (2, E_t)          sparse COO format
      E_t  = number of directed edges on day t
             Per node: 50 top + 50 bottom + 75 random = 175 undirected
             → ≤ 500 × 175 × 2 directed (with dedup)

  edge_attr       : (E_t, 5)
      [ρ_base, |ρ_base|, sign_indicator, same_sector, same_subind]

  edge_type       : (E_t,)            long tensor ∈ {0, 1, 2}
      0 = neg (bottom 1/3), 1 = mid, 2 = pos (top 1/3)

  edge_weight     : (E_t,)            float = ρ_base ∈ [-1, 1]

  target_z_resid  : (E_t,)            Δz = z_future − z_base  (Fisher-z space)
  baseline_z      : (E_t,)            z_base = atanh(ρ_base)

  node_mask       : (N_t,)            bool — which slots are valid stocks
  stock_ids       : (N_t,)            integer stock identifier for each node
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# PyTorch Geometric for graph batching
try:
    from torch_geometric.data import Data, Batch
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install via: "
        "pip install torch-geometric"
    )

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THGNNConfig


# ═══════════════════════════════════════════════════════════════════════════
# Helper: rolling z-score normalisation (60-day window)
# ═══════════════════════════════════════════════════════════════════════════
def rolling_zscore(
    arr: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Apply rolling z-score normalisation along axis 0 (time).

    Parameters
    ----------
    arr : np.ndarray, shape (T, F)
        Raw feature matrix for a single stock across T trading days.
    window : int
        Rolling window length (default 60 per paper).

    Returns
    -------
    np.ndarray, shape (T, F) — NaN for the first `window-1` rows.
    """
    T, F = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float32)
    for t in range(window, T + 1):
        block = arr[t - window : t]
        mu = block.mean(axis=0)
        sigma = block.std(axis=0) + 1e-8
        out[t - 1] = (arr[t - 1] - mu) / sigma
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Helper: edge construction per day t
# ═══════════════════════════════════════════════════════════════════════════
def build_graph_edges(
    corr_matrix: np.ndarray,
    sector_codes: np.ndarray,
    subind_codes: np.ndarray,
    cfg: THGNNConfig,
    rng: np.random.Generator | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Build undirected edge set from the 30-day rolling correlation matrix.

    For each node i we select:
      • top-50 most positive correlations
      • bottom-50 most negative correlations
      • 75 random from the [0.2, 0.8] correlation percentile band

    Parameters
    ----------
    corr_matrix : (N, N) pairwise rolling 30-day correlations.
    sector_codes : (N,)  integer GICS sector code per stock.
    subind_codes : (N,)  integer GICS sub-industry code per stock.
    cfg : THGNNConfig

    Returns
    -------
    dict with keys: edge_index, edge_attr, edge_type, edge_weight, baseline_z
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    N = corr_matrix.shape[0]
    src_list, dst_list = [], []
    attr_list, type_list, weight_list = [], [], []

    # Compute global tertile thresholds for edge-type classification
    upper_tri = corr_matrix[np.triu_indices(N, k=1)]
    upper_tri = upper_tri[~np.isnan(upper_tri)]
    if len(upper_tri) == 0:
        tercile_lo, tercile_hi = -0.33, 0.33
    else:
        tercile_lo, tercile_hi = np.percentile(upper_tri, [33.33, 66.67])

    for i in range(N):
        row = corr_matrix[i].copy()
        row[i] = np.nan  # exclude self-loop

        valid_mask = ~np.isnan(row)
        valid_idx = np.where(valid_mask)[0]
        valid_corr = row[valid_idx]

        if len(valid_idx) == 0:
            continue

        sorted_order = np.argsort(valid_corr)
        sorted_idx = valid_idx[sorted_order]

        # Top-K (most positive)
        top_k = sorted_idx[-cfg.top_k_corr :] if len(sorted_idx) >= cfg.top_k_corr else sorted_idx
        # Bottom-K (most negative)
        bot_k = sorted_idx[: cfg.bot_k_corr] if len(sorted_idx) >= cfg.bot_k_corr else sorted_idx

        # Mid-range random sampling: [0.2, 0.8] percentile of this node's corrs
        pct_lo, pct_hi = np.percentile(valid_corr, [20, 80])
        mid_mask = (valid_corr >= pct_lo) & (valid_corr <= pct_hi)
        mid_candidates = valid_idx[mid_mask]
        already_selected = set(top_k.tolist()) | set(bot_k.tolist())
        mid_candidates = np.array([j for j in mid_candidates if j not in already_selected])
        n_mid = min(cfg.rand_mid_k, len(mid_candidates))
        if n_mid > 0:
            mid_k = rng.choice(mid_candidates, size=n_mid, replace=False)
        else:
            mid_k = np.array([], dtype=np.int64)

        neighbours = np.concatenate([top_k, bot_k, mid_k]).astype(np.int64)
        neighbours = np.unique(neighbours)

        for j in neighbours:
            rho = corr_matrix[i, j]
            sign_ind = 0.0 if rho > 0 else 1.0
            same_sector = float(sector_codes[i] == sector_codes[j])
            same_subind = float(subind_codes[i] == subind_codes[j])

            # Edge type: neg=0, mid=1, pos=2
            if rho <= tercile_lo:
                etype = 0
            elif rho >= tercile_hi:
                etype = 2
            else:
                etype = 1

            src_list.append(i)
            dst_list.append(j)
            attr_list.append([rho, abs(rho), sign_ind, same_sector, same_subind])
            type_list.append(etype)
            weight_list.append(rho)

    if len(src_list) == 0:
        return dict(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, cfg.num_edge_attr, dtype=torch.float32),
            edge_type=torch.zeros(0, dtype=torch.long),
            edge_weight=torch.zeros(0, dtype=torch.float32),
            baseline_z=torch.zeros(0, dtype=torch.float32),
        )

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)        # (2, E)
    edge_attr = torch.tensor(attr_list, dtype=torch.float32)                 # (E, 5)
    edge_type = torch.tensor(type_list, dtype=torch.long)                    # (E,)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)             # (E,)
    baseline_z = torch.atanh(edge_weight.clamp(-0.9999, 0.9999))            # (E,)

    return dict(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        edge_weight=edge_weight,
        baseline_z=baseline_z,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Dataset
# ═══════════════════════════════════════════════════════════════════════════
class THGNNDataset(Dataset):
    """
    One sample = one trading-day graph snapshot.

    Expected raw data layout (to be provided by user's data pipeline):
    ─────────────────────────────────────────────────────────────────────
    features     : dict[stock_id → np.ndarray of shape (T_total, F)]
                   Raw (un-normalised) features per stock across all dates.
    dates        : list[str] of length T_total — sorted trading dates.
    sector_map   : dict[stock_id → int]   GICS sector code
    subind_map   : dict[stock_id → int]   GICS sub-industry code
    returns      : dict[stock_id → np.ndarray of shape (T_total,)]
                   Daily returns for rolling-correlation computation.
    """

    def __init__(
        self,
        features: Dict[int, np.ndarray],
        dates: List[str],
        sector_map: Dict[int, int],
        subind_map: Dict[int, int],
        returns: Dict[int, np.ndarray],
        cfg: THGNNConfig = THGNNConfig(),
        date_range: Optional[Tuple[str, str]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.features = features          # raw features per stock
        self.dates = dates
        self.sector_map = sector_map
        self.subind_map = subind_map
        self.returns = returns

        self.all_stock_ids = sorted(features.keys())
        for sid in self.all_stock_ids:
            if sid not in self.returns:
                raise KeyError(f"Missing returns for stock_id={sid}")
            if self.features[sid].shape[0] != len(self.dates):
                raise ValueError(
                    f"Feature length mismatch for stock_id={sid}: "
                    f"{self.features[sid].shape[0]} vs {len(self.dates)} dates."
                )
            if self.returns[sid].shape[0] != len(self.dates):
                raise ValueError(
                    f"Return length mismatch for stock_id={sid}: "
                    f"{self.returns[sid].shape[0]} vs {len(self.dates)} dates."
                )

        # Pre-compute normalised features (rolling 60-day z-score)
        self.norm_features: Dict[int, np.ndarray] = {}
        for sid, feat in self.features.items():
            self.norm_features[sid] = rolling_zscore(feat, cfg.rolling_zscore_window)

        # Determine eligible target dates for this split.
        # A valid target day must have:
        #   1. enough warm-up / eligibility history
        #   2. a full future forecast horizon available
        #   3. if date_range is provided, that future horizon must stay inside the split
        #
        # Without this guard, the last `forecast_horizon` days of a split either
        # get fake zero targets or leak future information across train/val splits.
        min_start = cfg.rolling_zscore_window - 1 + cfg.max_gap_days
        max_target = len(dates) - cfg.forecast_horizon - 1

        if date_range is not None:
            start, end = date_range
            split_end_idx = max(
                (i for i, d in enumerate(dates) if d <= end),
                default=-1,
            )
            max_target = min(max_target, split_end_idx - cfg.forecast_horizon)
            self.date_indices = [
                i for i, d in enumerate(dates)
                if start <= d <= end and min_start <= i <= max_target
            ]
        else:
            self.date_indices = list(range(min_start, max_target + 1))

        # Filter out dates where too few stocks are eligible to form a
        # meaningful correlation graph / supervision target.
        self.date_indices = [
            i for i in self.date_indices
            if len(self._eligible_stocks(i)) >= 2
        ]

    # ── helpers ────────────────────────────────────────────────────────
    def _eligible_stocks(self, t_idx: int) -> List[int]:
        """
        A stock is eligible on day t if it has ≥30 valid feature days
        within the last 33 trading days (Section 4.2.1).
        """
        cfg = self.cfg
        eligible = []
        window_start = max(0, t_idx - cfg.max_gap_days + 1)
        for sid in self.all_stock_ids:
            feat = self.norm_features[sid]
            if feat.shape[0] <= t_idx:
                continue
            window = feat[window_start : t_idx + 1]
            valid_days = np.sum(~np.isnan(window[:, 0]))  # check first feature col
            if valid_days >= cfg.min_trading_days:
                eligible.append(sid)
        return eligible

    def _compute_rolling_corr(
        self, stock_ids: List[int], t_idx: int
    ) -> np.ndarray:
        """
        Compute the N×N 30-day rolling pairwise correlation matrix
        from daily returns ending at date index t_idx.
        """
        cfg = self.cfg
        N = len(stock_ids)
        start = max(0, t_idx - cfg.rolling_corr_window + 1)
        ret_matrix = np.full((cfg.rolling_corr_window, N), np.nan, dtype=np.float32)
        for col, sid in enumerate(stock_ids):
            r = self.returns[sid]
            L = min(cfg.rolling_corr_window, t_idx - start + 1)
            ret_matrix[-L:, col] = r[start : t_idx + 1][-L:]

        # Pairwise Pearson correlation via numpy (handles NaN gracefully)
        ret_clean = np.nan_to_num(ret_matrix, nan=0.0)
        means = ret_clean.mean(axis=0, keepdims=True)
        centered = ret_clean - means
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        std = np.sqrt(np.diag(cov)).clip(min=1e-8)
        corr = cov / np.outer(std, std)
        return np.clip(corr, -1.0, 1.0)

    def _compute_future_corr(
        self, stock_ids: List[int], t_idx: int
    ) -> Optional[np.ndarray]:
        """
        Compute the 10-day *future* pairwise correlation using
        cumulative returns from t+1 to t+10 (Section 4.1).
        Returns None if insufficient future data.
        """
        cfg = self.cfg
        h = cfg.forecast_horizon
        end_idx = t_idx + h
        if end_idx >= len(self.dates):
            return None

        N = len(stock_ids)
        ret_matrix = np.full((h, N), np.nan, dtype=np.float32)
        for col, sid in enumerate(stock_ids):
            r = self.returns[sid]
            if r.shape[0] > end_idx:
                ret_matrix[:, col] = r[t_idx + 1 : end_idx + 1]

        ret_t = torch.from_numpy(np.nan_to_num(ret_matrix, 0.0))
        centered = ret_t - ret_t.mean(0, keepdim=True)
        cov = centered.T @ centered / max(h - 1, 1)
        std = torch.sqrt(torch.diagonal(cov).clamp(min=1e-8))
        corr = cov / (std.unsqueeze(1) * std.unsqueeze(0))
        return corr.clamp(-1, 1).numpy()

    # ── Dataset interface ─────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.date_indices)

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a PyG Data object for trading day t = self.date_indices[idx].

        Shapes inside the Data object:
            x             : (N_t, L=30, F=37)    node feature sequences
            edge_index    : (2, E_t)              COO edges
            edge_attr     : (E_t, 5)              edge attributes
            edge_type     : (E_t,)                {0,1,2}
            edge_weight   : (E_t,)                ρ_base
            baseline_z    : (E_t,)                atanh(ρ_base)
            target_z_resid: (E_t,)                Δz target (Fisher-z)
            num_nodes     : int
        """
        cfg = self.cfg
        t_idx = self.date_indices[idx]

        # 1. Determine eligible stocks
        stock_ids = self._eligible_stocks(t_idx)
        N = len(stock_ids)

        # 2. Assemble node feature sequences: (N, L, F)
        node_features = np.zeros((N, cfg.seq_len, cfg.num_features), dtype=np.float32)
        for i, sid in enumerate(stock_ids):
            feat = self.norm_features[sid]
            start = t_idx - cfg.seq_len + 1
            seq = feat[start : t_idx + 1]          # (L, F) ideally
            valid_len = seq.shape[0]
            # Zero-fill any NaN
            seq = np.nan_to_num(seq, 0.0)
            node_features[i, -valid_len:] = seq[-valid_len:]

        x = torch.from_numpy(node_features)         # (N, 30, 37)

        # 3. Compute rolling 30-day correlation → edge construction
        sector_codes = np.array([self.sector_map.get(s, 0) for s in stock_ids])
        subind_codes = np.array([self.subind_map.get(s, 0) for s in stock_ids])
        corr_base = self._compute_rolling_corr(stock_ids, t_idx)     # (N, N)
        rng = np.random.default_rng(cfg.seed + int(t_idx))

        edge_data = build_graph_edges(
            corr_base, sector_codes, subind_codes, cfg, rng
        )

        # 4. Compute target: Δz = z_future − z_base  per edge
        corr_future = self._compute_future_corr(stock_ids, t_idx)
        E = edge_data["edge_index"].shape[1]

        if corr_future is not None:
            z_future_mat = np.arctanh(np.clip(corr_future, -0.9999, 0.9999))
            z_base_mat = np.arctanh(np.clip(corr_base, -0.9999, 0.9999))
            src = edge_data["edge_index"][0].numpy()
            dst = edge_data["edge_index"][1].numpy()
            target_z_resid = torch.tensor(
                z_future_mat[src, dst] - z_base_mat[src, dst],
                dtype=torch.float32,
            )  # (E,)
        else:
            # Defensive fallback. Valid target dates are filtered in __init__,
            # so this branch should be unreachable in normal use.
            target_z_resid = torch.zeros(E, dtype=torch.float32)

        # 5. Pack into PyG Data object
        data = Data(
            x=x,                                      # (N, 30, 37)
            edge_index=edge_data["edge_index"],        # (2, E)
            edge_attr=edge_data["edge_attr"],          # (E, 5)
            edge_type=edge_data["edge_type"],          # (E,)
            edge_weight=edge_data["edge_weight"],      # (E,)
            baseline_z=edge_data["baseline_z"],        # (E,)
            target_z_resid=target_z_resid,             # (E,)
            num_nodes=N,
        )
        # Store metadata (not tensors — for debugging / interpretability)
        data.stock_ids = stock_ids
        data.date = self.dates[t_idx]

        return data


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════
def build_dataloader(
    dataset: THGNNDataset,
    cfg: THGNNConfig,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wraps the dataset in a PyG-compatible DataLoader that uses
    `torch_geometric.data.Batch` to collate variable-size graphs.

    Batch size = 3 (physical); gradient accumulation of 6 happens in
    the training loop for an effective batch of 18.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    return PyGDataLoader(
        dataset,
        batch_size=cfg.batch_size,     # 3
        shuffle=shuffle,
        num_workers=0,                 # set >0 for multiprocess
        drop_last=False,
    )
