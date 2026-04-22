from __future__ import annotations

__all__ = [
    "TemporalEncoder",
    "RelationalEncoder",
    "EdgeAwareGATLayer",
    "ExpertPredictionHeads",
    "ExpertMLP",
    "THGNN",
]


def __getattr__(name: str):
    if name in {"TemporalEncoder"}:
        from .temporal_encoder import TemporalEncoder

        return TemporalEncoder
    if name in {"RelationalEncoder", "EdgeAwareGATLayer"}:
        from .relational_encoder import RelationalEncoder, EdgeAwareGATLayer

        return {
            "RelationalEncoder": RelationalEncoder,
            "EdgeAwareGATLayer": EdgeAwareGATLayer,
        }[name]
    if name in {"ExpertPredictionHeads", "ExpertMLP"}:
        from .expert_heads import ExpertPredictionHeads, ExpertMLP

        return {
            "ExpertPredictionHeads": ExpertPredictionHeads,
            "ExpertMLP": ExpertMLP,
        }[name]
    if name == "THGNN":
        from .thgnn import THGNN

        return THGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
