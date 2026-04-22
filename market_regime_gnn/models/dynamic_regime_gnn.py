from __future__ import annotations

from .._loader import load_legacy_module

_dynamic_model = load_legacy_module("models.dynamic_regime_gnn")

NodeFeatureEncoder = _dynamic_model.NodeFeatureEncoder
SpatialRGCN = _dynamic_model.SpatialRGCN
TemporalLSTM = _dynamic_model.TemporalLSTM
TemporalTransformer = _dynamic_model.TemporalTransformer
RegimeClassifierHead = _dynamic_model.RegimeClassifierHead
TransitionLogitHead = _dynamic_model.TransitionLogitHead
DynamicRegimeGNN = _dynamic_model.DynamicRegimeGNN

__all__ = [
    "NodeFeatureEncoder",
    "SpatialRGCN",
    "TemporalLSTM",
    "TemporalTransformer",
    "RegimeClassifierHead",
    "TransitionLogitHead",
    "DynamicRegimeGNN",
]
