# evaluate.py — placeholder evaluator
import argparse
"RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
"MAE": float(mean_absolute_error(y_test, y_pred)),
"MSE": float(mean_squared_error(y_test, y_pred)),
"NSE": float(nse(y_test.values, y_pred)),
"KGE": float(kge(y_test.values, y_pred)),
}


import os
os.makedirs("results", exist_ok=True)
with open("results/metrics.json", "w", encoding="utf-8") as f:
json.dump(results, f, indent=2)


# Save per-row test predictions (optionally with GridID/Year/Month)
pred_df = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred})
if meta_te is not None:
pred_df = pd.concat([meta_te.reset_index(drop=True), pred_df], axis=1)
pred_df.to_csv("results/test_predictions.csv", index=False)


# Per-Grid metrics if GridID exists
id_col = config["data"].get("id_col")
if id_col and id_col in pred_df.columns:
rows = []
for gid, g in pred_df.groupby(id_col):
yt = g["y_test"].to_numpy()
yp = g["y_pred"].to_numpy()
if len(yt) == 0:
continue
rows.append({
id_col: gid,
"MSE": float(np.mean((yt - yp) ** 2)),
"RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
"NSE": float(nse(yt, yp)),
"Pbias": float((np.mean(yp - yt) / (np.mean(yt) if np.mean(yt) != 0 else 1e-12)) * 100.0),
"KGE": float(kge(yt, yp)),
"R2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
})
pd.DataFrame(rows).to_csv("results/metrics_by_grid.csv", index=False)
