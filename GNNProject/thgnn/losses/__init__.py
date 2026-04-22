from __future__ import annotations

__all__ = ["THGNNLoss", "soft_histogram", "make_bin_centers"]


def __getattr__(name: str):
    if name in __all__:
        from .loss import THGNNLoss, soft_histogram, make_bin_centers

        return {
            "THGNNLoss": THGNNLoss,
            "soft_histogram": soft_histogram,
            "make_bin_centers": make_bin_centers,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
