# data.py — utilities to load/download data (placeholder)
from pathlib import Path

def get_sample_path(root="data/sample"):
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p
