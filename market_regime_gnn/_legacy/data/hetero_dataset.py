"""
Heterogeneous Temporal Graph Dataset
======================================
Constructs sequences of T=30 daily HeteroData graph snapshots for the
Multi-Relational Dynamic GNN for Market Regime Detection.

Architecture context:
────────────────────
    Each sample = a SEQUENCE of T=30 consecutive daily heterogeneous graphs
    + a scalar regime_label ∈ {0,1,2,3} (Bull/Crash/Liquidity/Stress)
    + a binary transition_label ∈ {0,1} (stress onset in 5–20 days)

    Each daily graph snapshot is a torch_geometric.data.HeteroData with:
        • Node type: "stock"
            x        : (N, F)           — per-stock features for that day
        • Edge types: ("stock", "correlation", "stock")
                      ("stock", "etf_cohold", "stock")
                      ("stock", "supply_chain", "stock")
            edge_index : (2, E_r)       — COO edges per relation
            edge_attr  : (E_r, D_e)     — edge attributes per relation

Graph construction per snapshot:
─────────────────────────────────
    Relation 0 — correlation:
        30-day rolling pairwise return correlation → top-K / bottom-K
        per node. Edge weight = ρ.

    Relation 1 — etf_cohold:
        ETF co-holding overlap (Jaccard index of ETF membership sets).
        Edge weight = Jaccard similarity. Constructed from an ETF
        holdings matrix (stocks × ETFs), updated quarterly.

    Relation 2 — supply_chain:
        Production-network edges from BEA I-O tables or Compustat
        segments. Binary (connected or not), edge weight = 1.0.

Key design decision — batching sequences:
──────────────────────────────────────────
    __getitem__ returns a dict:
        {
            "snapshots": list[HeteroData] of length T=30,
            "regime_label": int,           # 0–3
            "transition_label": int,       # 0 or 1
            "date": str,                   # the target date (day T)
        }

    A custom collate_fn stacks these into batch-friendly format:
        {
            "snapshots": list[list[HeteroData]]  — (B, T)
            "regime_label": LongTensor (B,)
            "transition_label": FloatTensor (B,)
        }

    The model processes each timestep's batch independently through the
    spatial encoder, then sequences the graph-level embeddings through
    the temporal aggregator.

Shapes reference:
─────────────────
    per snapshot:
        stock.x            : (N_t, 37)        — node features
        (stock, corr, stock).edge_index : (2, E_corr)
        (stock, corr, stock).edge_attr  : (E_corr, 4)
        (stock, etf, stock).edge_index  : (2, E_etf)
        (stock, etf, stock).edge_attr   : (E_etf, 4)
        (stock, sc, stock).edge_index   : (2, E_sc)
        (stock, sc, stock).edge_attr    : (E_sc, 4)

    per sample:
        snapshots    : T=30 HeteroData objects
        regime_label : scalar int ∈ {0,1,2,3}
        transition_label : scalar int ∈ {0,1}
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import HeteroData, Batch
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install via: "
        "pip install torch-geometric"
    )

try:
    from ..config import RegimeConfig
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import RegimeConfig


# ═══════════════════════════════════════════════════════════════════════════
# Constants: edge type canonical names for HeteroData
# ═══════════════════════════════════════════════════════════════════════════
EDGE_TYPES = [
    ("stock", "correlation", "stock"),
    ("stock", "etf_cohold", "stock"),
    ("stock", "supply_chain", "stock"),
]

# Integer relation IDs matching config.relation_names order
REL_CORRELATION = 0
REL_ETF_COHOLD = 1
REL_SUPPLY_CHAIN = 2


# ═══════════════════════════════════════════════════════════════════════════
# Helper: rolling z-score normalisation
# ═══════════════════════════════════════════════════════════════════════════
def rolling_zscore(arr: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Rolling z-score along axis 0 (time).

    Parameters
    ----------
    arr : (T, F) raw features for one stock.
    window : rolling window length.

    Returns
    -------
    (T, F) — NaN for the first `window-1` rows.
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
# Helper: build edges for one relation on one day
# ═══════════════════════════════════════════════════════════════════════════
def build_correlation_edges(
    corr_matrix: np.ndarray,
    sector_codes: np.ndarray,
    subind_codes: np.ndarray,
    cfg: RegimeConfig,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """
    Build directed correlation edges: top-K positive + bottom-K negative
    per node from 30-day rolling correlation matrix.

    Returns dict with edge_index (2, E), edge_attr (E, 4).
    edge_attr = [ρ, |ρ|, same_sector, same_subind]
    """
    N = corr_matrix.shape[0]
    src_list, dst_list, attr_list = [], [], []

    for i in range(N):
        row = corr_matrix[i].copy()
        row[i] = np.nan  # no self-loops

        valid_mask = ~np.isnan(row)
        valid_idx = np.where(valid_mask)[0]
        valid_corr = row[valid_idx]

        if len(valid_idx) == 0:
            continue

        sorted_order = np.argsort(valid_corr)
        sorted_idx = valid_idx[sorted_order]

        # Top-K (most positive)
        top_k = sorted_idx[-cfg.corr_top_k:] if len(sorted_idx) >= cfg.corr_top_k else sorted_idx
        # Bottom-K (most negative)
        bot_k = sorted_idx[:cfg.corr_bot_k] if len(sorted_idx) >= cfg.corr_bot_k else sorted_idx

        neighbours = np.unique(np.concatenate([top_k, bot_k]))

        for j in neighbours:
            rho = corr_matrix[i, j]
            same_sec = float(sector_codes[i] == sector_codes[j])
            same_sub = float(subind_codes[i] == subind_codes[j])
            src_list.append(i)
            dst_list.append(j)
            attr_list.append([rho, abs(rho), same_sec, same_sub])

    if len(src_list) == 0:
        return dict(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, cfg.edge_attr_dim, dtype=torch.float32),
        )

    return dict(
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_attr=torch.tensor(attr_list, dtype=torch.float32),
    )


def build_etf_cohold_edges(
    etf_holdings: Optional[np.ndarray],
    sector_codes: np.ndarray,
    subind_codes: np.ndarray,
    stock_ids: List[int],
    cfg: RegimeConfig,
) -> Dict[str, torch.Tensor]:
    """
    Build ETF co-holding edges from a holdings matrix.

    Parameters
    ----------
    etf_holdings : (N, M) binary matrix — stock i is held by ETF j.
                   If None, creates synthetic edges based on sector overlap.
    sector_codes : (N,) integer sector codes.
    subind_codes : (N,) integer sub-industry codes.
    stock_ids    : list of stock IDs (for mapping).
    cfg          : RegimeConfig.

    Returns
    -------
    dict with edge_index (2, E), edge_attr (E, 4).
    edge_attr = [jaccard, |jaccard|, same_sector, same_subind]
    """
    N = len(stock_ids)

    if etf_holdings is not None:
        # Compute Jaccard similarity between all pairs
        # holdings: (N, M) binary
        intersection = etf_holdings @ etf_holdings.T     # (N, N)
        row_sums = etf_holdings.sum(axis=1, keepdims=True)  # (N, 1)
        union = row_sums + row_sums.T - intersection      # (N, N)
        jaccard = intersection / np.clip(union, 1, None)   # (N, N)
    else:
        # Synthetic fallback: sector-based similarity
        # Same sector → high overlap, same sub-industry → higher
        jaccard = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                sim = 0.0
                if sector_codes[i] == sector_codes[j]:
                    sim = 0.4
                if subind_codes[i] == subind_codes[j]:
                    sim = 0.7
                jaccard[i, j] = sim
                jaccard[j, i] = sim

    src_list, dst_list, attr_list = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if jaccard[i, j] >= cfg.etf_cohold_threshold:
                same_sec = float(sector_codes[i] == sector_codes[j])
                same_sub = float(subind_codes[i] == subind_codes[j])
                src_list.append(i)
                dst_list.append(j)
                attr_list.append([jaccard[i, j], jaccard[i, j], same_sec, same_sub])

    if len(src_list) == 0:
        return dict(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, cfg.edge_attr_dim, dtype=torch.float32),
        )

    return dict(
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_attr=torch.tensor(attr_list, dtype=torch.float32),
    )


def build_supply_chain_edges(
    supply_chain_adj: Optional[np.ndarray],
    sector_codes: np.ndarray,
    subind_codes: np.ndarray,
    stock_ids: List[int],
    cfg: RegimeConfig,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """
    Build supply-chain edges from a production-network adjacency matrix.

    Parameters
    ----------
    supply_chain_adj : (N, N) binary adjacency — 1 if i supplies to j.
                       If None, creates sparse synthetic edges.
    sector_codes     : (N,) integer sector codes.
    subind_codes     : (N,) integer sub-industry codes.
    stock_ids        : list of stock IDs.
    cfg              : RegimeConfig.
    rng              : numpy random generator.

    Returns
    -------
    dict with edge_index (2, E), edge_attr (E, 4).
    edge_attr = [1.0, 1.0, same_sector, same_subind]  (binary edges)
    """
    N = len(stock_ids)

    if supply_chain_adj is not None:
        adj = supply_chain_adj
    else:
        # Synthetic fallback: sparse random edges (~3 per node)
        # Biased towards cross-sector links (supply chains span sectors)
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            n_links = rng.integers(1, 5)  # 1–4 supply-chain links
            candidates = [j for j in range(N) if j != i]
            if len(candidates) > 0:
                chosen = rng.choice(candidates, size=min(n_links, len(candidates)),
                                     replace=False)
                for j in chosen:
                    adj[i, j] = 1.0

    src_list, dst_list, attr_list = [], [], []
    for i in range(N):
        for j in range(N):
            if adj[i, j] > 0.5:
                same_sec = float(sector_codes[i] == sector_codes[j])
                same_sub = float(subind_codes[i] == subind_codes[j])
                src_list.append(i)
                dst_list.append(j)
                attr_list.append([1.0, 1.0, same_sec, same_sub])

    if len(src_list) == 0:
        return dict(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, cfg.edge_attr_dim, dtype=torch.float32),
        )

    return dict(
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_attr=torch.tensor(attr_list, dtype=torch.float32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper: rolling correlation matrix
# ═══════════════════════════════════════════════════════════════════════════
def compute_rolling_corr(
    returns: Dict[int, np.ndarray],
    stock_ids: List[int],
    t_idx: int,
    window: int = 30,
) -> np.ndarray:
    """
    Compute N×N pairwise 30-day rolling correlation from daily returns.

    Parameters
    ----------
    returns   : dict mapping stock_id → (T_total,) return array.
    stock_ids : list of stock IDs defining node order.
    t_idx     : current date index (inclusive).
    window    : rolling window length.

    Returns
    -------
    (N, N) correlation matrix.
    """
    N = len(stock_ids)
    start = max(0, t_idx - window + 1)
    ret_matrix = np.full((window, N), 0.0, dtype=np.float32)

    for col, sid in enumerate(stock_ids):
        r = returns[sid]
        L = min(window, t_idx - start + 1)
        if r.shape[0] > t_idx:
            ret_matrix[-L:, col] = r[start : t_idx + 1][-L:]

    means = ret_matrix.mean(axis=0, keepdims=True)
    centered = ret_matrix - means
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    std = np.sqrt(np.diag(cov)).clip(min=1e-8)
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Main Dataset
# ═══════════════════════════════════════════════════════════════════════════
class RegimeDataset(Dataset):
    """
    Temporal sequence of heterogeneous graph snapshots for regime detection.

    Each sample = (T=30 HeteroData snapshots, regime_label, transition_label).

    Expected raw data layout (provided by data pipeline):
    ──────────────────────────────────────────────────────
    features        : dict[stock_id → np.ndarray (T_total, F)]
    dates           : list[str] of length T_total, sorted
    sector_map      : dict[stock_id → int]    GICS sector code
    subind_map      : dict[stock_id → int]    GICS sub-industry code
    returns         : dict[stock_id → np.ndarray (T_total,)]
    regime_labels   : np.ndarray (T_total,)   int ∈ {0,1,2,3}
    transition_labels : np.ndarray (T_total,) int ∈ {0,1}
    etf_holdings    : Optional[np.ndarray (N, M)]   binary ETF membership
    supply_chain_adj: Optional[np.ndarray (N, N)]   binary adjacency

    The regime_labels and transition_labels are aligned to the dates array:
        regime_labels[t] = regime class on day t
        transition_labels[t] = 1 if stress onset occurs in days [t+5, t+20]
    """

    def __init__(
        self,
        features: Dict[int, np.ndarray],
        dates: List[str],
        sector_map: Dict[int, int],
        subind_map: Dict[int, int],
        returns: Dict[int, np.ndarray],
        regime_labels: np.ndarray,
        transition_labels: np.ndarray,
        cfg: Optional[RegimeConfig] = None,
        date_range: Optional[Tuple[str, str]] = None,
        etf_holdings: Optional[np.ndarray] = None,
        supply_chain_adj: Optional[np.ndarray] = None,
    ):
        super().__init__()
        cfg = RegimeConfig() if cfg is None else cfg
        self.cfg = cfg
        self.features = features
        self.dates = dates
        self.sector_map = sector_map
        self.subind_map = subind_map
        self.returns = returns
        self.regime_labels = regime_labels
        self.transition_labels = transition_labels
        self.etf_holdings = (
            np.asarray(etf_holdings) if etf_holdings is not None else None
        )
        self.supply_chain_adj = (
            np.asarray(supply_chain_adj) if supply_chain_adj is not None else None
        )
        self.all_stock_ids = sorted(features.keys())
        self.stock_id_to_pos = {
            sid: idx for idx, sid in enumerate(self.all_stock_ids)
        }
        self._eligible_stock_cache: Dict[int, Tuple[int, ...]] = {}
        self._snapshot_cache: Dict[Tuple[int, Tuple[int, ...]], HeteroData] = {}

        if len(self.regime_labels) != len(self.dates):
            raise ValueError(
                "regime_labels must be aligned 1:1 with dates."
            )
        if len(self.transition_labels) != len(self.dates):
            raise ValueError(
                "transition_labels must be aligned 1:1 with dates."
            )
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

        n_total = len(self.all_stock_ids)
        if self.etf_holdings is not None:
            if self.etf_holdings.ndim != 2 or self.etf_holdings.shape[0] != n_total:
                raise ValueError(
                    "etf_holdings must have shape (num_stocks, num_etfs) aligned "
                    "with sorted feature keys."
                )
        if self.supply_chain_adj is not None:
            if self.supply_chain_adj.shape != (n_total, n_total):
                raise ValueError(
                    "supply_chain_adj must have shape (num_stocks, num_stocks) "
                    "aligned with sorted feature keys."
                )

        # Pre-compute z-scored features
        self.norm_features: Dict[int, np.ndarray] = {}
        for sid, feat in self.features.items():
            self.norm_features[sid] = rolling_zscore(feat, cfg.rolling_zscore_window)

        # Determine valid target dates.
        # A valid day needs:
        #   1. enough history for rolling z-score + sequence lookback
        #   2. a full transition horizon available
        #   3. if a split range is provided, that full horizon must remain
        #      inside the split to avoid train/val label leakage.
        #
        # The earliest valid target is when the first snapshot in the
        # sequence lands exactly on the first z-scored day:
        #   t_target - (seq_len - 1) >= rolling_zscore_window - 1
        min_start = cfg.rolling_zscore_window + cfg.seq_len - 2
        max_target = len(dates) - cfg.transition_horizon_max - 1

        if date_range is not None:
            start_str, end_str = date_range
            split_end_idx = max(
                (i for i, d in enumerate(dates) if d <= end_str),
                default=-1,
            )
            max_target = min(max_target, split_end_idx - cfg.transition_horizon_max)
            self.date_indices = [
                i for i, d in enumerate(dates)
                if start_str <= d <= end_str and min_start <= i <= max_target
            ]
        else:
            self.date_indices = list(range(min_start, max_target + 1))

        # Exclude dates that would yield empty or trivial graphs.
        self.date_indices = [
            i for i in self.date_indices
            if len(self._get_eligible_stocks(i)) >= 2
        ]

    def __len__(self) -> int:
        return len(self.date_indices)

    def _get_eligible_stocks(self, t_idx: int) -> Tuple[int, ...]:
        """Stocks with valid data at time t_idx."""
        cached = self._eligible_stock_cache.get(t_idx)
        if cached is not None:
            return cached

        eligible = []
        for sid in self.all_stock_ids:
            feat = self.norm_features[sid]
            if feat.shape[0] <= t_idx:
                continue
            if not np.isnan(feat[t_idx, 0]):
                eligible.append(sid)
        cached = tuple(eligible)
        self._eligible_stock_cache[t_idx] = cached
        return cached

    def _get_snapshot(
        self,
        t_idx: int,
        stock_ids: Tuple[int, ...],
    ) -> HeteroData:
        """
        Reuse previously built daily snapshots across overlapping sequences.

        Samples in adjacent windows share most of their constituent days, so
        memoizing the CPU snapshot avoids rebuilding identical graph structure
        on every __getitem__ call. The training pipeline now batches each
        timestep on CPU and only moves the PyG Batch to the accelerator, so
        the cached base object can be returned directly without mutation.
        """
        cache_key = (t_idx, stock_ids)
        cached = self._snapshot_cache.get(cache_key)
        if cached is None:
            cached = self._build_snapshot(t_idx, list(stock_ids))
            self._snapshot_cache[cache_key] = cached
        return cached

    def _build_snapshot(self, t_idx: int, stock_ids: List[int]) -> HeteroData:
        """
        Build one HeteroData graph snapshot for day t_idx.

        Parameters
        ----------
        t_idx     : date index into self.dates.
        stock_ids : list of eligible stock IDs (defines node ordering).

        Returns
        -------
        HeteroData with node features and 3 edge types.

        Shapes:
            stock.x                             : (N, F)
            (stock, correlation, stock).edge_index : (2, E_corr)
            (stock, correlation, stock).edge_attr  : (E_corr, 4)
            (stock, etf_cohold, stock).edge_index  : (2, E_etf)
            (stock, etf_cohold, stock).edge_attr   : (E_etf, 4)
            (stock, supply_chain, stock).edge_index : (2, E_sc)
            (stock, supply_chain, stock).edge_attr  : (E_sc, 4)
        """
        cfg = self.cfg
        N = len(stock_ids)
        rng = np.random.default_rng(cfg.seed + int(t_idx))

        # ── Node features: use day t_idx's z-scored features ────────────
        node_features = np.zeros((N, cfg.num_features), dtype=np.float32)
        for i, sid in enumerate(stock_ids):
            feat = self.norm_features[sid]
            if t_idx < feat.shape[0]:
                row = feat[t_idx]
                node_features[i] = np.nan_to_num(row, 0.0)

        data = HeteroData()
        data["stock"].x = torch.from_numpy(node_features)  # (N, 37)
        data["stock"].num_nodes = N

        # ── Sector / sub-industry codes ─────────────────────────────────
        sector_codes = np.array([self.sector_map.get(s, 0) for s in stock_ids])
        subind_codes = np.array([self.subind_map.get(s, 0) for s in stock_ids])
        matrix_idx = np.array(
            [self.stock_id_to_pos[sid] for sid in stock_ids],
            dtype=np.int64,
        )

        # ── Relation 0: Correlation edges ───────────────────────────────
        corr_matrix = compute_rolling_corr(
            self.returns, stock_ids, t_idx, cfg.corr_window,
        )
        corr_edges = build_correlation_edges(
            corr_matrix, sector_codes, subind_codes, cfg, rng,
        )
        data["stock", "correlation", "stock"].edge_index = corr_edges["edge_index"]
        data["stock", "correlation", "stock"].edge_attr = corr_edges["edge_attr"]

        # ── Relation 1: ETF co-holding edges ────────────────────────────
        etf_holdings = None
        if self.etf_holdings is not None:
            etf_holdings = self.etf_holdings[matrix_idx]

        etf_edges = build_etf_cohold_edges(
            etf_holdings, sector_codes, subind_codes, stock_ids, cfg,
        )
        data["stock", "etf_cohold", "stock"].edge_index = etf_edges["edge_index"]
        data["stock", "etf_cohold", "stock"].edge_attr = etf_edges["edge_attr"]

        # ── Relation 2: Supply-chain edges ──────────────────────────────
        supply_chain_adj = None
        if self.supply_chain_adj is not None:
            supply_chain_adj = self.supply_chain_adj[np.ix_(matrix_idx, matrix_idx)]

        sc_edges = build_supply_chain_edges(
            supply_chain_adj, sector_codes, subind_codes, stock_ids,
            cfg, rng,
        )
        data["stock", "supply_chain", "stock"].edge_index = sc_edges["edge_index"]
        data["stock", "supply_chain", "stock"].edge_attr = sc_edges["edge_attr"]

        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a temporal sequence of T=30 HeteroData graph snapshots
        plus scalar labels.

        Returns
        -------
        dict:
            "snapshots"        : list[HeteroData] of length T=30
            "regime_label"     : int ∈ {0,1,2,3}
            "transition_label" : int ∈ {0,1}
            "date"             : str — target date (day T of the sequence)
        """
        cfg = self.cfg
        t_target = self.date_indices[idx]  # the "current" (last) day

        # Use consistent stock universe across the T=30 window
        # (determined by eligibility on the target day)
        stock_ids = self._get_eligible_stocks(t_target)

        # Build T=30 consecutive daily snapshots ending at t_target
        snapshots: List[HeteroData] = []
        for offset in range(cfg.seq_len - 1, -1, -1):
            # t goes from (t_target - 29) to t_target
            t_idx = t_target - offset
            snapshot = self._get_snapshot(t_idx, stock_ids)
            snapshots.append(snapshot)

        assert len(snapshots) == cfg.seq_len, \
            f"Expected {cfg.seq_len} snapshots, got {len(snapshots)}"

        # Labels aligned to the target date
        regime_label = int(self.regime_labels[t_target])
        transition_label = int(self.transition_labels[t_target])

        return {
            "snapshots": snapshots,
            "regime_label": regime_label,
            "transition_label": transition_label,
            "date": self.dates[t_target],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Custom collate function for DataLoader
# ═══════════════════════════════════════════════════════════════════════════
def regime_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate for RegimeDataset.

    Converts a list of B samples into a batch-friendly dict:
        "snapshots"        : list[list[HeteroData]] — shape (T, B)
                             Transposed so each entry is one timestep's
                             batch of B graphs (for efficient PyG Batch.from_data_list)
        "regime_label"     : LongTensor (B,)
        "transition_label" : FloatTensor (B,)

    The model can then iterate over T timesteps, batching each
    timestep's B graphs with Batch.from_data_list.
    """
    B = len(batch)
    T = len(batch[0]["snapshots"])

    # Transpose (B, T) → (T, B) so each timestep has B graphs
    snapshots_by_time: List[List[HeteroData]] = []
    for t in range(T):
        timestep_graphs = [batch[b]["snapshots"][t] for b in range(B)]
        snapshots_by_time.append(timestep_graphs)

    regime_labels = torch.tensor(
        [b["regime_label"] for b in batch], dtype=torch.long
    )
    transition_labels = torch.tensor(
        [b["transition_label"] for b in batch], dtype=torch.float32
    )

    return {
        "snapshots": snapshots_by_time,          # (T, B) list of lists
        "regime_label": regime_labels,            # (B,)
        "transition_label": transition_labels,    # (B,)
    }


def build_regime_dataloader(
    dataset: RegimeDataset,
    cfg: RegimeConfig,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader with the custom collate function.

    Each batch yields:
        snapshots        : (T=30, B) list of HeteroData
        regime_label     : (B,) LongTensor
        transition_label : (B,) FloatTensor
    """
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=regime_collate_fn,
        drop_last=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generator (for testing)
# ═══════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(
    n_stocks: int = 30,
    n_days: int = 200,
    n_features: int = 37,
    seed: int = 42,
) -> Tuple[
    Dict[int, np.ndarray],   # features
    List[str],                # dates
    Dict[int, int],           # sector_map
    Dict[int, int],           # subind_map
    Dict[int, np.ndarray],   # returns
    np.ndarray,               # regime_labels
    np.ndarray,               # transition_labels
]:
    """
    Generate synthetic data for smoke testing the dataset.

    Returns all arrays needed for RegimeDataset constructor.
    """
    import datetime

    rng = np.random.default_rng(seed)

    # Dates
    base = datetime.date(2020, 1, 1)
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]

    # Per-stock data
    features, returns_dict, sector_map, subind_map = {}, {}, {}, {}
    for sid in range(n_stocks):
        features[sid] = rng.standard_normal((n_days, n_features)).astype(np.float32)
        returns_dict[sid] = (rng.standard_normal(n_days) * 0.02).astype(np.float32)
        sector_map[sid] = int(rng.integers(0, 11))
        subind_map[sid] = int(rng.integers(0, 50))

    # Regime labels: semi-realistic cycle
    # Bull=0 (50%), Crash=1 (15%), Liquidity=2 (15%), Stress=3 (20%)
    regime_labels = np.zeros(n_days, dtype=np.int64)
    probs = [0.50, 0.15, 0.15, 0.20]
    for t in range(n_days):
        regime_labels[t] = rng.choice(4, p=probs)

    # Transition labels: 1 if stress in next 5–20 days
    transition_labels = np.zeros(n_days, dtype=np.int64)
    for t in range(n_days):
        future = regime_labels[t + 5 : min(t + 21, n_days)]
        if 3 in future:  # Stress class
            transition_labels[t] = 1

    return features, dates, sector_map, subind_map, returns_dict, regime_labels, transition_labels


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  RegimeDataset — Smoke Test")
    print("=" * 70)

    cfg = RegimeConfig()
    cfg.seq_len = 30
    cfg.corr_top_k = 5
    cfg.corr_bot_k = 5

    # Generate synthetic data
    (features, dates, sector_map, subind_map,
     returns_dict, regime_labels, transition_labels) = generate_synthetic_data(
        n_stocks=20, n_days=200, seed=42,
    )

    print(f"\nSynthetic data:")
    print(f"  Stocks: {len(features)}, Days: {len(dates)}")
    print(f"  Regime dist: {np.bincount(regime_labels, minlength=4)}")
    print(f"  Transition dist: {np.bincount(transition_labels, minlength=2)}")

    # Build dataset
    ds = RegimeDataset(
        features=features, dates=dates,
        sector_map=sector_map, subind_map=subind_map,
        returns=returns_dict,
        regime_labels=regime_labels,
        transition_labels=transition_labels,
        cfg=cfg,
    )
    print(f"\nDataset samples: {len(ds)}")

    # Check a single sample
    sample = ds[0]
    print(f"\nSample[0]:")
    print(f"  # snapshots      : {len(sample['snapshots'])}")
    print(f"  regime_label     : {sample['regime_label']}")
    print(f"  transition_label : {sample['transition_label']}")
    print(f"  date             : {sample['date']}")

    snap = sample["snapshots"][0]
    print(f"\n  Snapshot[0] (first day):")
    print(f"    stock.x           : {tuple(snap['stock'].x.shape)}")
    print(f"    stock.num_nodes   : {snap['stock'].num_nodes}")
    for et_name in EDGE_TYPES:
        ei = snap[et_name].edge_index
        ea = snap[et_name].edge_attr
        print(f"    {et_name[1]:15s} : edges={ei.shape[1]:4d}, attr={tuple(ea.shape)}")

    snap_last = sample["snapshots"][-1]
    print(f"\n  Snapshot[29] (last day):")
    print(f"    stock.x           : {tuple(snap_last['stock'].x.shape)}")
    for et_name in EDGE_TYPES:
        ei = snap_last[et_name].edge_index
        print(f"    {et_name[1]:15s} : edges={ei.shape[1]:4d}")

    # Check dataloader
    loader = build_regime_dataloader(ds, cfg, shuffle=False)
    batch = next(iter(loader))

    print(f"\nBatch (batch_size={cfg.batch_size}):")
    print(f"  snapshots       : {len(batch['snapshots'])} timesteps × "
          f"{len(batch['snapshots'][0])} graphs per timestep")
    print(f"  regime_label    : {tuple(batch['regime_label'].shape)} — {batch['regime_label']}")
    print(f"  transition_label: {tuple(batch['transition_label'].shape)} — {batch['transition_label']}")

    # Batch the first timestep using PyG
    batched_t0 = Batch.from_data_list(batch["snapshots"][0])
    print(f"\n  PyG Batch (t=0):")
    print(f"    stock.x        : {tuple(batched_t0['stock'].x.shape)}")
    print(f"    stock.batch    : {tuple(batched_t0['stock'].batch.shape)}")
    for et_name in EDGE_TYPES:
        if et_name in batched_t0.edge_types:
            ei = batched_t0[et_name].edge_index
            print(f"    {et_name[1]:15s} : edges={ei.shape[1]}")

    print(f"\n{'=' * 70}")
    print(f"  RegimeDataset smoke test PASSED ✓")
    print(f"{'=' * 70}")
