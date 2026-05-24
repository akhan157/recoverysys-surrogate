"""
Train the surrogate model on generated simulation data.

Usage:
    python -m surrogate.train --data data/train.parquet --epochs 100
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from surrogate.model import INPUT_FEATURES, OUTPUT_FEATURES, SurrogateNet


def load_data(path: Path):
    df = pd.read_parquet(path)
    X = df[INPUT_FEATURES].values.astype(np.float32)
    y = df[OUTPUT_FEATURES].values.astype(np.float32)
    return X, y


def train(data_path: Path, model_path: Path, epochs: int, batch_size: int, lr: float):
    X, y = load_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train = torch.tensor(x_scaler.transform(X_train))
    X_val   = torch.tensor(x_scaler.transform(X_val))
    y_train = torch.tensor(y_scaler.transform(y_train))
    y_val   = torch.tensor(y_scaler.transform(y_val))

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "x_scaler_mean": x_scaler.mean_,
                "x_scaler_scale": x_scaler.scale_,
                "y_scaler_mean": y_scaler.mean_,
                "y_scaler_scale": y_scaler.scale_,
            }, model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val_loss:.4f}")

    print(f"\nBest model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=Path,  default=Path("data/train.parquet"))
    parser.add_argument("--model",      type=Path,  default=Path("models/surrogate_v1.pt"))
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, args.model, args.epochs, args.batch_size, args.lr)
