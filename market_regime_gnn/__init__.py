"""
Import-friendly wrapper package for the Dynamic Regime GNN prototype.

This package exposes the main project under a normal import path and
bundles the legacy implementation inside `market_regime_gnn._legacy`
so editable installs and built wheels behave the same way.
"""

from .config import RegimeConfig

__all__ = ["RegimeConfig"]
