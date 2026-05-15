from pathlib import Path


def _is_reusable_output_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def resolve_output_dir(dataset_root: Path, env_name: str) -> Path:
    base_dir = Path(dataset_root) / env_name
    if not base_dir.exists() or _is_reusable_output_dir(base_dir):
        return base_dir

    version = 2
    while True:
        candidate = Path(dataset_root) / f"{env_name}_v{version}"
        if not candidate.exists() or _is_reusable_output_dir(candidate):
            return candidate
        version += 1
