from __future__ import annotations

from .._loader import load_legacy_module

_label_generator = load_legacy_module("data.label_generator")

REGIME_BULL = _label_generator.REGIME_BULL
REGIME_CRASH = _label_generator.REGIME_CRASH
REGIME_LIQUIDITY = _label_generator.REGIME_LIQUIDITY
REGIME_STRESS = _label_generator.REGIME_STRESS
REGIME_NAMES = _label_generator.REGIME_NAMES

compute_avg_cross_correlation = _label_generator.compute_avg_cross_correlation
compute_rolling_metrics = _label_generator.compute_rolling_metrics
classify_regimes = _label_generator.classify_regimes
compute_transition_labels = _label_generator.compute_transition_labels
generate_market_labels = _label_generator.generate_market_labels
generate_labels_from_yahoo = _label_generator.generate_labels_from_yahoo

__all__ = [
    "REGIME_BULL",
    "REGIME_CRASH",
    "REGIME_LIQUIDITY",
    "REGIME_STRESS",
    "REGIME_NAMES",
    "compute_avg_cross_correlation",
    "compute_rolling_metrics",
    "classify_regimes",
    "compute_transition_labels",
    "generate_market_labels",
    "generate_labels_from_yahoo",
]
