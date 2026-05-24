"""
Feedforward neural network surrogate for the RecoverySys flight simulator.

Architecture: 10 inputs -> [256, 256, 128] hidden -> 6 outputs
All hidden layers use BatchNorm + ReLU + Dropout(0.1).
"""

import torch
import torch.nn as nn


INPUT_FEATURES = [
    "rocket_mass_kg",
    "motor_total_impulse_ns",
    "burn_time_s",
    "airframe_od_in",
    "cd",
    "main_chute_diameter_in",
    "drogue_chute_diameter_in",
    "main_deploy_alt_ft",
    "wind_speed_mph",
    "wind_direction_deg",
]

OUTPUT_FEATURES = [
    "apogee_ft",
    "max_mach",
    "drogue_fps",
    "main_fps",
    "drift_ft",
    "p95_drift_ft",
]


class SurrogateNet(nn.Module):
    def __init__(self, hidden: list[int] = [256, 256, 128], dropout: float = 0.1):
        super().__init__()
        dims = [len(INPUT_FEATURES)] + hidden + [len(OUTPUT_FEATURES)]
        layers = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
