"""Locate a saved run without selecting an example or training a new model."""
from pathlib import Path


def resolve_saved_run(value=None):
    if value:return Path(value).expanduser().resolve()
    base=Path.home()/'Arm26_NN_Results';pointer=base/'LATEST_COMPLETE_PIPELINE.txt'
    if pointer.exists():
        path=Path(pointer.read_text().strip()).resolve()
        if (path/'policy.pt').exists() and (path/'history.csv').exists():return path
    if base.exists():
        candidates=[p for p in base.iterdir() if p.is_dir() and (p/'policy.pt').exists() and (p/'history.csv').exists() and (p/'tracking'/'comparison.json').exists()]
        if candidates:return max(candidates,key=lambda p:p.stat().st_mtime).resolve()
    raise FileNotFoundError('No saved trained result found. Supply --run-folder with your completed result folder.')
