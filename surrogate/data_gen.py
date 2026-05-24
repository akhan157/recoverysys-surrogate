"""
Sample the recoverysys-engine simulation API to generate training data.

Usage:
    python -m surrogate.data_gen --samples 5000 --out data/train.parquet
"""

import argparse
import json
import random
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

ENGINE_URL = "http://localhost:8000/api/simulate"

# Sampling ranges derived from realistic high-power rocketry configurations
PARAM_RANGES = {
    "rocket_mass_kg":           (1.0,   8.0),
    "motor_total_impulse_ns":   (200,   5000),
    "burn_time_s":              (0.8,   4.0),
    "airframe_od_in":           (2.0,   5.0),
    "cd":                       (0.35,  0.75),
    "main_chute_diameter_in":   (36,    96),
    "drogue_chute_diameter_in": (12,    24),
    "wind_speed_mph":           (0,     25),
    "wind_direction_deg":       (0,     359),
    "main_deploy_alt_ft":       (400,   1000),
}

OUTPUT_FIELDS = [
    "apogee_ft", "max_mach", "drogue_fps", "main_fps",
    "drift_ft", "p95_drift_ft",
]


def sample_params() -> dict:
    p = {k: random.uniform(*v) for k, v in PARAM_RANGES.items()}
    return {
        "rocket_mass_kg":         round(p["rocket_mass_kg"], 3),
        "motor_total_impulse_ns": round(p["motor_total_impulse_ns"], 1),
        "burn_time_s":            round(p["burn_time_s"], 2),
        "airframe_od_in":         round(p["airframe_od_in"], 2),
        "cd":                     round(p["cd"], 3),
        "main_chute":    {"diameter_in": round(p["main_chute_diameter_in"]),  "cd": 2.2},
        "drogue_chute":  {"diameter_in": round(p["drogue_chute_diameter_in"]), "cd": 1.5},
        "main_deploy_alt_ft": round(p["main_deploy_alt_ft"]),
        "wind_layers": [
            {
                "altitude_ft":   0,
                "speed_mph":     round(p["wind_speed_mph"], 1),
                "direction_deg": round(p["wind_direction_deg"], 1),
            }
        ],
        "monte_carlo": {"enabled": True, "iterations": 200},
    }


def run_simulation(params: dict, client: httpx.Client) -> dict | None:
    try:
        resp = client.post(ENGINE_URL, json=params, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return {
            "apogee_ft":    result["ascent"]["apogee_ft"],
            "max_mach":     result["ascent"]["max_mach"],
            "drogue_fps":   result["descent"]["drogue_fps"],
            "main_fps":     result["descent"]["main_fps"],
            "drift_ft":     result["drift"]["distance_ft"],
            "p95_drift_ft": result["monte_carlo"]["p95_drift_ft"],
        }
    except Exception:
        return None


def flatten_params(params: dict) -> dict:
    return {
        "rocket_mass_kg":           params["rocket_mass_kg"],
        "motor_total_impulse_ns":   params["motor_total_impulse_ns"],
        "burn_time_s":              params["burn_time_s"],
        "airframe_od_in":           params["airframe_od_in"],
        "cd":                       params["cd"],
        "main_chute_diameter_in":   params["main_chute"]["diameter_in"],
        "drogue_chute_diameter_in": params["drogue_chute"]["diameter_in"],
        "main_deploy_alt_ft":       params["main_deploy_alt_ft"],
        "wind_speed_mph":           params["wind_layers"][0]["speed_mph"],
        "wind_direction_deg":       params["wind_layers"][0]["direction_deg"],
    }


def generate(n_samples: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = 0

    with httpx.Client() as client:
        for _ in tqdm(range(n_samples), desc="Sampling"):
            params = sample_params()
            outputs = run_simulation(params, client)
            if outputs is None:
                failed += 1
                continue
            row = flatten_params(params) | outputs
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}  ({failed} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--out",     type=Path, default=Path("data/train.parquet"))
    args = parser.parse_args()
    generate(args.samples, args.out)
