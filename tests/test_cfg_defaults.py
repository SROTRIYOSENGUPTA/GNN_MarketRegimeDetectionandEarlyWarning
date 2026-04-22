import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch_geometric")

from GNNProject.thgnn.data.dataset import THGNNDataset
from GNNProject.thgnn.losses.loss import THGNNLoss
from GNNProject.thgnn.models.expert_heads import ExpertPredictionHeads
from GNNProject.thgnn.models.relational_encoder import RelationalEncoder
from GNNProject.thgnn.models.temporal_encoder import TemporalEncoder
from GNNProject.thgnn.models.thgnn import THGNN
from GNNProject.thgnn.train import Trainer as THGNNTrainer
from market_regime_gnn._legacy.data.hetero_dataset import RegimeDataset
from market_regime_gnn._legacy.models.dynamic_regime_gnn import DynamicRegimeGNN
from market_regime_gnn._legacy.train import Trainer as RegimeTrainer


def _cfg_default(callable_obj):
    return inspect.signature(callable_obj).parameters["cfg"].default


def test_cfg_parameters_use_none_sentinels():
    assert _cfg_default(THGNNDataset.__init__) is None
    assert _cfg_default(THGNNLoss.__init__) is None
    assert _cfg_default(TemporalEncoder.__init__) is None
    assert _cfg_default(RelationalEncoder.__init__) is None
    assert _cfg_default(ExpertPredictionHeads.__init__) is None
    assert _cfg_default(THGNN.__init__) is None
    assert _cfg_default(THGNNTrainer.__init__) is None
    assert _cfg_default(RegimeDataset.__init__) is None
    assert _cfg_default(DynamicRegimeGNN.__init__) is None
    assert _cfg_default(RegimeTrainer.__init__) is None
