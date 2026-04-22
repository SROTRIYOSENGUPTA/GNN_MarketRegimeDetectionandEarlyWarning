from __future__ import annotations

from ._loader import load_legacy_module

def _legacy():
    return load_legacy_module("run_real_data")


def fetch_real_data(*args, **kwargs):
    return _legacy().fetch_real_data(*args, **kwargs)


def main(*args, **kwargs):
    return _legacy().main(*args, **kwargs)


def build_split_date_ranges(start: str, end: str, train_cutoff: str):
    return _legacy().build_split_date_ranges(start, end, train_cutoff)


def validate_split_sample_counts(
    train_samples: int,
    val_samples: int,
    train_range: tuple[str, str],
    val_range: tuple[str, str],
):
    return _legacy().validate_split_sample_counts(
        train_samples, val_samples, train_range, val_range
    )


def resolve_device(device):
    return _legacy().resolve_device(device)


def validate_runtime_args(**kwargs):
    return _legacy().validate_runtime_args(**kwargs)


def build_arg_parser():
    return _legacy().build_arg_parser()


def cli(argv: list[str] | None = None) -> int:
    return _legacy().cli(argv)


__all__ = [
    "fetch_real_data",
    "main",
    "build_split_date_ranges",
    "validate_split_sample_counts",
    "resolve_device",
    "validate_runtime_args",
    "build_arg_parser",
    "cli",
]


if __name__ == "__main__":
    raise SystemExit(cli())
