# interpret.py — placeholder SHAP step
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/baseline.yaml")
args = parser.parse_args()


with open(args.config, "r", encoding="utf-8") as f:
config = yaml.safe_load(f)


X_all, y_all, features, target, meta = load_all_features(config)
model = joblib.load("models/xgb_model.joblib")


# SHAP for tree models
explainer = shap.TreeExplainer(model)


# Sampling to control memory/time (configurable)
sample_n = int(config.get("interpret", {}).get("sample_n", 20000))
if sample_n and sample_n < len(X_all):
X_used = X_all.sample(n=sample_n, random_state=config.get("split", {}).get("seed", 42))
else:
X_used = X_all


shap_values = explainer(X_used)


# Save global importance (mean |SHAP|)
mean_abs = np.abs(shap_values.values).mean(axis=0)
imp = pd.Series(mean_abs, index=X_used.columns).sort_values(ascending=False)
imp.to_json("results/shap_importance.json", indent=2)


# Beeswarm summary plot
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=min(26, X_used.shape[1]), show=False)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=300)
plt.close()


# Optional dependence plots for selected features
dep_feats = config.get("interpret", {}).get("dependence_features", ["PRE", "WYMI", "Tmax", "GDP", "Shrubland"])
present = [f for f in dep_feats if f in X_used.columns]
for ftr in present:
try:
plt.figure(figsize=(6, 5))
shap.plots.scatter(shap_values[:, ftr], show=False)
plt.tight_layout()
safe_name = ftr.replace("/", "_")
plt.savefig(f"results/shap_dependence_{safe_name}.png", dpi=300)
plt.close()
except Exception:
pass


# Optional: save full SHAP matrix (may be large)
if config.get("interpret", {}).get("save_values", False):
np.save("results/shap_values.npy", shap_values.values)
X_used.to_csv("results/shap_features_sample.csv", index=False)


# Also dump a small manifest
manifest = {
"n_rows_used": int(X_used.shape[0]),
"n_features": int(X_used.shape[1]),
"top5_features": imp.head(5).to_dict(),
}
with open("results/shap_manifest.json", "w", encoding="utf-8") as fp:
json.dump(manifest, fp, indent=2)
