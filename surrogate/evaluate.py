"""
Evaluate surrogate model accuracy vs. the physics engine.

Reports per-output MAE, MAPE, and R², plus inference speed comparison.

Usage:
    python -m surrogate.evaluate --model models/surrogate_v1.pt --data data/test.parquet
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from surrogate.model import INPUT_FEATURES, OUTPUT_FEATURES, SurrogateNet


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model = SurrogateNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate(model_path: Path, data_path: Path):
    model, ckpt = load_checkpoint(model_path)

    df = pd.read_parquet(data_path)
    X = df[INPUT_FEATURES].values.astype(np.float32)
    y_true = df[OUTPUT_FEATURES].values.astype(np.float32)

    x_mean  = ckpt["x_scaler_mean"].astype(np.float32)
    x_scale = ckpt["x_scaler_scale"].astype(np.float32)
    y_mean  = ckpt["y_scaler_mean"].astype(np.float32)
    y_scale = ckpt["y_scaler_scale"].astype(np.float32)

    X_scaled = torch.tensor((X - x_mean) / x_scale)

    t0 = time.perf_counter()
    with torch.no_grad():
        y_pred_scaled = model(X_scaled).numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    y_pred = y_pred_scaled * y_scale + y_mean

    print(f"\n{'Output':<20} {'MAE':>12} {'MAPE %':>10} {'R²':>8}")
    print("-" * 54)
    for i, col in enumerate(OUTPUT_FEATURES):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-6))) * 100
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{col:<20} {mae:>12.2f} {mape:>10.2f} {r2:>8.4f}")

    n = len(X)
    print(f"\nInference: {elapsed_ms:.1f} ms for {n} samples  ({elapsed_ms/n*1000:.1f} µs/sample)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/surrogate_v1.pt"))
    parser.add_argument("--data",  type=Path, default=Path("data/test.parquet"))
    args = parser.parse_args()
    evaluate(args.model, args.data)
