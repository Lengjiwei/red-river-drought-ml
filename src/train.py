# train.py — placeholder training loop
# src/train.py — XGBoost 
from __future__ import annotations
import json, numpy as np, yaml, joblib
from pathlib import Path
from model import build_xgb
from data import load_frame, make_splits
from sklearn.metrics import mean_squared_error

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    df = load_frame(cfg["data"]["file"])
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = make_splits(
        df=df, method=cfg["split"]["method"], cfg_data=cfg["data"], cfg_split=cfg["split"]
    )

    model = build_xgb(cfg["xgb"])
    if Xva is not None and len(Xva) > 0:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr, verbose=False)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/xgb_model.joblib")
    save_json({
        "features": cfg["data"]["features"],
        "target": cfg["data"]["target"],
        "time_col": cfg["data"]["time_col"],
        "id_col": cfg["data"]["id_col"],
        "split": cfg["split"],
    }, "models/model_meta.json")

    report = {}
    if Xva is not None and len(Xva) > 0:
        val_rmse = mean_squared_error(yva, model.predict(Xva), squared=False)
        report["val_rmse"] = float(val_rmse)
    save_json(report, "results/train_report.json")
    print("[train] done. model -> models/xgb_model.joblib")
