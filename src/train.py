# train.py — placeholder training loop
import json, random, numpy as np, torch
from pathlib import Path
from model import CNNLSTM

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

if __name__ == "__main__":
    set_seed(42)
    # Dummy data: [B, T, F] = [64, 12, 4]
    x = torch.randn(64, 12, 4)
    y = torch.randn(64)
    model = CNNLSTM(in_channels=4)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        optim.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward(); optim.step()
    Path("results").mkdir(exist_ok=True, parents=True)
    with open("results/metrics.json","w") as f:
        json.dump({"train_loss": float(loss.item())}, f, indent=2)
    print("Training done. Metrics saved to results/metrics.json")
