from __future__ import annotations

from ._loader import load_legacy_module

_train = load_legacy_module("train")

FocalCrossEntropyLoss = _train.FocalCrossEntropyLoss
Trainer = _train.Trainer
compute_metrics = _train.compute_metrics
move_hetero_to_device = _train.move_hetero_to_device
move_snapshots_to_device = _train.move_snapshots_to_device

__all__ = [
    "FocalCrossEntropyLoss",
    "Trainer",
    "compute_metrics",
    "move_hetero_to_device",
    "move_snapshots_to_device",
]
