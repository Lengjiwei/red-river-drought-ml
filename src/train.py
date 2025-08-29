# train.py — placeholder training loop
import argparse
)
return float(cv['test-rmse-mean'].min())


trials = Trials()
best_idxs = fmin(fn=score, space=space, algo=tpe.suggest, max_evals=200, trials=trials, rstate=np.random.default_rng(seed))
best = space_eval(space, best_idxs)
return best




if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/baseline.yaml")
args = parser.parse_args()


with open(args.config, "r", encoding="utf-8") as f:
config = yaml.safe_load(f)


X_train, y_train, X_test, y_test = load_data(config)


# Optional hyperopt
if config.get("model", {}).get("hyperopt", {}).get("enable", False):
seed = config.get("split", {}).get("seed", 42)
best = run_hyperopt(X_train, y_train, seed)
# merge back to params
cfg_params = config.setdefault("model", {}).setdefault("params", {})
cfg_params.update({k: v for k, v in best.items() if k in [
'max_depth','colsample_bytree','min_child_weight','subsample','learning_rate','gamma'
]})
# Persist best params for reference
with open("results/best_params.json", "w", encoding="utf-8") as fp:
json.dump(cfg_params, fp, indent=2)


model = build_model(config)
model.fit(X_train, y_train)


# Save model
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.joblib")
