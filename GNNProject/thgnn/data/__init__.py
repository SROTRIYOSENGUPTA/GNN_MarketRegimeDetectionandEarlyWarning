from __future__ import annotations

__all__ = [
    "THGNNDataset",
    "build_dataloader",
    "build_graph_edges",
]


def __getattr__(name: str):
    if name in __all__:
        from .dataset import THGNNDataset, build_dataloader, build_graph_edges

        namespace = {
            "THGNNDataset": THGNNDataset,
            "build_dataloader": build_dataloader,
            "build_graph_edges": build_graph_edges,
        }
        return namespace[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
