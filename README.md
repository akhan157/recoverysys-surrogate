# surrogate-recoverysys

A neural network surrogate model for high-power rocketry flight simulation.

The physics simulation in [recoverysys-engine](https://github.com/akhan157/recoverysys-engine) is accurate but runs ODE integration on every call. This project trains a neural network on simulation outputs so that — given rocket parameters and wind conditions — predictions run in microseconds instead of milliseconds, with quantified accuracy loss.

**Status: in progress**

---

## What it does

1. **Data generation** — samples the recoverysys-engine Monte Carlo pipeline across a range of rocket configurations and wind conditions, producing a labeled dataset of inputs → simulation outputs
2. **Surrogate model** — trains a feedforward neural network (PyTorch) to map inputs directly to outputs, bypassing the ODE solver
3. **Evaluation** — compares surrogate predictions against the physics engine across held-out configurations: accuracy, speed, and failure modes

## Inputs / outputs

| Input | Description |
|-------|-------------|
| `rocket_mass_kg` | Total rocket mass |
| `motor_total_impulse_ns` | Motor total impulse |
| `burn_time_s` | Motor burn time |
| `airframe_od_in` | Airframe outer diameter |
| `cd` | Drag coefficient |
| `main_chute_diameter_in` | Main parachute diameter |
| `drogue_chute_diameter_in` | Drogue parachute diameter |
| `wind_speed_mph` | Mean wind speed |
| `wind_direction_deg` | Wind direction |

| Output | Description |
|--------|-------------|
| `apogee_ft` | Peak altitude |
| `max_mach` | Max Mach number during ascent |
| `drogue_fps` | Drogue descent rate |
| `main_fps` | Main descent rate |
| `drift_ft` | Lateral drift distance |
| `p95_drift_ft` | 95th-percentile drift (Monte Carlo) |

## Stack

- Python 3.11+
- PyTorch — neural network training
- SciPy / NumPy — data generation via recoverysys-engine
- scikit-learn — preprocessing, baseline comparison
- Jupyter — exploratory notebooks

## Project structure

```
surrogate/
  data_gen.py    sample the physics engine, save training data
  model.py       neural network architecture
  train.py       training loop
  evaluate.py    accuracy vs. speed benchmarks
notebooks/
  01_data_exploration.ipynb
  02_model_training.ipynb
  03_results.ipynb
data/            generated datasets (gitignored)
models/          saved weights (gitignored)
tests/
```

## Setup

```bash
pip install -e ".[dev]"
```

Generate training data (requires recoverysys-engine running locally):

```bash
python -m surrogate.data_gen --samples 5000 --out data/train.parquet
```

Train:

```bash
python -m surrogate.train --data data/train.parquet --epochs 100
```

Evaluate:

```bash
python -m surrogate.evaluate --model models/surrogate_v1.pt --data data/test.parquet
```

## Related

- [recoverysys-engine](https://github.com/akhan157/recoverysys-engine) — the physics simulation this surrogate approximates
- [RecoverySys](https://github.com/akhan157/RecoverySys) — the frontend tool built on that engine
