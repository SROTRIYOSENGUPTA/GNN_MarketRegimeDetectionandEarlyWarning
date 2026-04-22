from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PACKAGED_LEGACY_ROOT = Path(__file__).resolve().parent / "_legacy"
SOURCE_LEGACY_ROOT = (
    Path(__file__).resolve().parents[1]
    / "GNNsMarketRegimeDetection&Early-Warning"
)
LEGACY_PACKAGE = "market_regime_gnn._legacy"


def _load_package(module_name: str, package_path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    init_path = package_path / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_path)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for package {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_from_source_tree(relative_module: str) -> ModuleType:
    _load_package(LEGACY_PACKAGE, SOURCE_LEGACY_ROOT)

    parts = relative_module.split(".")
    parent_name = LEGACY_PACKAGE
    parent_path = SOURCE_LEGACY_ROOT

    for part in parts[:-1]:
        parent_name = f"{parent_name}.{part}"
        parent_path = parent_path / part
        _load_package(parent_name, parent_path)

    module_name = f"{LEGACY_PACKAGE}.{relative_module}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = SOURCE_LEGACY_ROOT.joinpath(*parts).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for module {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_module(relative_module: str) -> ModuleType:
    """
    Load one legacy module under a stable alias package.

    Example:
        load_legacy_module("data.label_generator")
        -> market_regime_gnn._legacy.data.label_generator
    """
    module_name = f"{LEGACY_PACKAGE}.{relative_module}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        if not PACKAGED_LEGACY_ROOT.exists() and not SOURCE_LEGACY_ROOT.exists():
            raise ImportError(
                f"Neither packaged legacy modules nor source tree exists for {module_name}"
            ) from exc
        return _load_from_source_tree(relative_module)
