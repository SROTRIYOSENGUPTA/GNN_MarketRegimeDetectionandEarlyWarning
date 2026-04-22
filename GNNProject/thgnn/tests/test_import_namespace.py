"""
Import regression test for THGNN package namespacing.
"""

from pathlib import Path
import sys

import pytest

pytest.importorskip("torch_geometric")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def test_thgnn_package_imports_do_not_pollute_bare_module_names():
    sys.modules.pop("config", None)
    sys.modules.pop("models", None)
    sys.modules.pop("data", None)

    import GNNProject.thgnn.config as config_module
    import GNNProject.thgnn.models.thgnn as thgnn_module

    assert config_module.__name__ == "GNNProject.thgnn.config"
    assert thgnn_module.__name__ == "GNNProject.thgnn.models.thgnn"
    assert "config" not in sys.modules
    assert "models" not in sys.modules
    assert "data" not in sys.modules
