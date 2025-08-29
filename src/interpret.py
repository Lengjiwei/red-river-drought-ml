# interpret.py — placeholder SHAP step
from __future__ import annotations
import json, yaml, joblib, numpy as np
from pathlib import Path
from data import load_frame, make_splits
import shap
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg  = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model = joblib.load("models/xgb_model.joblib")
    feat_names = cfg["data"]["features"]

    df = load_frame(cfg["data"]["file"])
    (_, _, _), (Xva, yva, _), (Xte, yte, _) = make_splits(df, cfg["data"], cfg["split"])

    Xexp = Xva if Xva is not None and len(Xva) > 0 else Xte
    if len(Xexp) > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xexp), size=2000, replace=False)
        Xexp = Xexp[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xexp)

    ensure_dir("results")
  
    imp = np.mean(np.abs(shap_values), axis=0)
    imp_dict = dict(zip(feat_names, imp.tolist()))
    with open("results/shap_importance.json","w",encoding="utf-8") as f:
        json.dump(imp_dict, f, indent=2)

    # summary plot
    plt.figure()
    shap.summary_plot(shap_values, Xexp, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=200)
    plt.close()

    print("[interpret] done. SHAP -> results/shap_importance.json, results/shap_summary.png")

