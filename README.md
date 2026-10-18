# DopplerPilot

Implementation of **Risk-aware pilot/data co-design for finite-blocklength grant-free URLLC under uncertain aging.**

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`prepare` samples Doppler-dependent correlations and random active load for offline policy design. `run` draws independent evaluation scenarios. All actions share blocklength, packet energy, peak powers and the feasible pilot grid. This is a link-surrogate simulation: conditional BLER comes from the finite-blocklength normal approximation.

```bash
python -m dopplerpilot prepare --config config.json --output runs/default
```

## Training and inference

```bash
python -m dopplerpilot fit --config config.json --output runs/default
python -m dopplerpilot run --config config.json --output runs/default
```

## Evaluation

```bash
python -m dopplerpilot evaluate --config config.json --output runs/default
python -m dopplerpilot plot --config config.json --output runs/default
```

## Outputs

`design.npz` holds optimization scenarios; `policy.csv` holds selected pilot/power actions; `packets.csv`, `metrics.csv` and `bler.png` contain held-out link-surrogate results.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, pandas 2.2.3, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
