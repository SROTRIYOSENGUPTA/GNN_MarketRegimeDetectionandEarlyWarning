from __future__ import annotations

from ._loader import load_legacy_module

_config = load_legacy_module("config")

RegimeConfig = _config.RegimeConfig

__all__ = ["RegimeConfig"]
