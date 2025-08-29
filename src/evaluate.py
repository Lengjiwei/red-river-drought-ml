# evaluate.py — placeholder evaluator
import json, numpy as np
from pathlib import Path

if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True, parents=True)
    metrics = {"rmse": float(np.random.rand()), "r": float(np.random.rand())}
    with open("results/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("Evaluation done. Metrics saved to results/metrics.json")
