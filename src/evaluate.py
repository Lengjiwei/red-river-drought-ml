# evaluate.py — placeholder evaluator
# src/evaluate.py 
from __future__ import annotations
import json, joblib, yaml, numpy as np
from pathlib import Path
from data import load_frame, make_splits
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def nse(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - num/den if den > 0 else float("nan")

def kge(y_true, y_pred):
    """Kling-Gupta Efficiency"""
    mu_o = np.mean(y_true)
    mu_s = np.mean(y_pred)
    std_o = np.std(y_true)
    std_s = np.std(y_pred)

    r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (float("nan"), None)
    alpha = std_s / std_o if std_o > 0 else float("nan")
    beta  = mu_s / mu_o if mu_o != 0 else float("nan")

    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg  = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model = joblib.load("models/xgb_model.joblib")

    df = load_frame(cfg["data"]["file"])
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = make_splits(df, cfg["data"], cfg["split"])

    def metrics(y, p):
        rmse = mean_squared_error(y, p, squared=False)
        mae  = mean_absolute_error(y, p)
        r2   = r2_score(y, p)
        r, _ = pearsonr(y, p) if len(y) > 1 else (float("nan"), None)
        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "R": float(r),
            "NSE": float(nse(y, p)),
            "KGE": float(kge(y, p))
        }

    out = {}
    if len(Xva) > 0:
        out["val"] = metrics(yva, model.predict(Xva))
    out["test"] = metrics(yte, model.predict(Xte))

    save_json(out, "results/metrics.json")
    print("[evaluate] done. results -> results/metrics.json")

