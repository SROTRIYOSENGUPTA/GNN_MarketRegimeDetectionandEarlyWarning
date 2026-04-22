import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from GNNProject.thgnn.run_real_data import resolve_device as resolve_thgnn_device
from market_regime_gnn.run_real_data import resolve_device as resolve_regime_device


RESOLVERS = [resolve_regime_device, resolve_thgnn_device]


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_resolve_device_accepts_cpu(resolver):
    assert resolver("cpu").type == "cpu"


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_resolve_device_rejects_invalid_name(resolver):
    with pytest.raises(ValueError, match="Invalid --device|Unsupported --device"):
        resolver("quantum")


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_resolve_device_validates_cuda_availability(resolver):
    if torch.cuda.is_available():
        assert resolver("cuda").type == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA device requested"):
            resolver("cuda")


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_resolve_device_validates_mps_availability(resolver):
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(
        mps_backend
        and getattr(mps_backend, "is_built", lambda: False)()
        and getattr(mps_backend, "is_available", lambda: False)()
    )

    if mps_available:
        assert resolver("mps").type == "mps"
    else:
        with pytest.raises(ValueError, match="MPS device requested"):
            resolver("mps")
