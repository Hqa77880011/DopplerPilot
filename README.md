# DopplerPilot

Implementation of **Risk-aware pilot/data co-design for finite-blocklength grant-free URLLC under uncertain aging.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

`prepare` samples Doppler-dependent correlations and random active load for offline policy design. `run` draws independent evaluation scenarios. All actions share blocklength, packet energy, peak powers and the feasible pilot grid. This is a link-surrogate simulation: conditional BLER comes from the finite-blocklength normal approximation.

## Workflow

```bash
python -m dopplerpilot prepare
python -m dopplerpilot fit
python -m dopplerpilot run
python -m dopplerpilot evaluate
python -m dopplerpilot plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`blocklength,bits,energy` define the packet budget. `pilots,fractions,fraction_points,pilot_peak,data_peak` define feasible complete allocations. `carrier_hz,separation_seconds,speed_std,speeds,loads` define uncertainty. `risk_weight,alpha` specify mean plus upper-tail CVaR. `reliability_threshold` fixes violation accounting. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`design.npz` contains optimization scenarios. `policy.csv` stores offline-selected pilot/power actions. `packets.csv` contains held-out surrogate errors, goodput and oracle agreement. `metrics.csv` reports mean BLER, CVaR BLER, violation probability, goodput, decision accuracy and lookup time. `bler.png` shows the mean-error curve.

## Implementation and controls

Exact enumeration minimizes the empirical criterion on the finite feasible grid. Pilot length changes estimation quality, remaining data length and data power together. CVaR integrates the correct fractional tail mass. The table is used online; timing includes the implemented pandas state lookup. Goodput uses payload bits per total packet channel use. The per-realization oracle is an evaluation diagnostic.

Controls: `DopplerPilot`, `mean-only`, `aging-blind`, `fixed`. The run command evaluates the listed controls together.

```bash
python -m dopplerpilot.compare --inputs "runs/*/metrics.csv" --reference DopplerPilot --metrics mean_bler cvar_bler violation goodput --unit seed
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

