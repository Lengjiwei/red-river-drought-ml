# data.py — utilities to load/download data (placeholder)
# src/data.py — 
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def load_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df

def ensure_datetime(df: pd.DataFrame, year_col: str, month_col: str, time_col: str = "date") -> pd.DataFrame:
    if time_col not in df.columns:
        yy = df[year_col].astype(int)
        mm = df[month_col].astype(int).clip(1, 12)
        df[time_col] = pd.to_datetime(dict(year=yy, month=mm, day=1))
    return df

def _per_cell_random_split(
    df: pd.DataFrame,
    id_col: str,
    features: List[str],
    target: str,
    train_ratio: float,
    test_ratio: float,
    val_within_train: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
):
    assert abs(train_ratio + test_ratio - 1.0) < 1e-8, "train_ratio + test_ratio 必须为 1.0"
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}

    missing = [c for c in features + [target, id_col] if c not in df.columns]
    if missing:
        raise ValueError(f"data：{missing}")

    for gid, g in df.groupby(id_col, sort=False):
        n = len(g)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        n_train_base = int(round(n * train_ratio))
        train_base_idx = idx[:n_train_base]
        test_idx = idx[n_train_base:]

        if val_within_train > 0:
            n_val = int(round(len(train_base_idx) * val_within_train))
            val_idx = train_base_idx[:n_val]
            train_idx = train_base_idx[n_val:]
        else:
            val_idx = np.array([], dtype=int)
            train_idx = train_base_idx

        parts["train"].append(g.iloc[train_idx])
        if len(val_idx) > 0:
            parts["val"].append(g.iloc[val_idx])
        parts["test"].append(g.iloc[test_idx])

    dtrain = pd.concat(parts["train"], axis=0, ignore_index=True) if parts["train"] else df.iloc[0:0]
    dval   = pd.concat(parts["val"],   axis=0, ignore_index=True) if parts["val"]   else df.iloc[0:0]
    dtest  = pd.concat(parts["test"],  axis=0, ignore_index=True) if parts["test"]  else df.iloc[0:0]

    def XY(d: pd.DataFrame):
        X = d[features].to_numpy()
        y = d[target].to_numpy()
        return X, y, d

    return XY(dtrain), XY(dval), XY(dtest)

def make_splits(
    df: pd.DataFrame,
    method: str,
    cfg_data: Dict,
    cfg_split: Dict,
):
    features = cfg_data["features"]
    target   = cfg_data["target"]

    year_col  = cfg_data.get("year_col", "Year")
    month_col = cfg_data.get("month_col", "Month")
    time_col  = cfg_data.get("time_col", "date")
    df = ensure_datetime(df, year_col=year_col, month_col=month_col, time_col=time_col)

    if method == "per_cell_random":
        return _per_cell_random_split(
            df=df,
            id_col=cfg_data["id_col"],
            features=features,
            target=target,
            train_ratio=float(cfg_split["ratios"]["train"]),
            test_ratio=float(cfg_split["ratios"]["test"]),
            val_within_train=float(cfg_split.get("val_within_train", 0.2)),
            shuffle=bool(cfg_split.get("shuffle", True)),
            seed=int(cfg_split.get("seed", 42)),
        )
    else:
        raise ValueError(f"Unsupported split method: {method}")

