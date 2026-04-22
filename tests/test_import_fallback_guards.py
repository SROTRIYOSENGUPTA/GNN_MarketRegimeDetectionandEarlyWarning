import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FILES_WITH_SCRIPT_FALLBACKS = [
    "GNNsMarketRegimeDetection&Early-Warning/data/hetero_dataset.py",
    "GNNsMarketRegimeDetection&Early-Warning/models/dynamic_regime_gnn.py",
    "GNNsMarketRegimeDetection&Early-Warning/train.py",
    "GNNsMarketRegimeDetection&Early-Warning/run_real_data.py",
    "GNNProject/thgnn/data/dataset.py",
    "GNNProject/thgnn/losses/loss.py",
    "GNNProject/thgnn/models/expert_heads.py",
    "GNNProject/thgnn/models/relational_encoder.py",
    "GNNProject/thgnn/models/temporal_encoder.py",
    "GNNProject/thgnn/models/thgnn.py",
    "GNNProject/thgnn/train.py",
    "GNNProject/thgnn/run_real_data.py",
]


def test_script_import_fallbacks_are_guarded():
    guard_pattern = re.compile(
        r"except ImportError as exc:\n\s+if __package__:\n\s+raise"
    )

    for rel_path in FILES_WITH_SCRIPT_FALLBACKS:
        text = (REPO_ROOT / rel_path).read_text()
        assert guard_pattern.search(text), f"missing guarded fallback in {rel_path}"
