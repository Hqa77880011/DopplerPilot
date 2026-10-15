import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import j0, ndtr


def tail_mean(values, alpha):
    n = len(values)
    mass = (1 - alpha) * n
    ordered = np.sort(values)[::-1]
    whole = int(np.floor(mass))
    fraction = mass - whole
    total = ordered[:whole].sum()
    if fraction > 0:
        total += fraction * ordered[whole]
    return total / mass


def actions(c):
    result = []
    for tau in c['pilots']:
        if not 0 < tau < c['blocklength']:
            raise ValueError('Pilot lengths must lie inside packet blocklength')
        for fraction in np.linspace(*c['fractions'], c['fraction_points']):
            pp = c['energy'] * fraction / tau
            pd = c['energy'] * (1 - fraction) / (c['blocklength'] - tau)
            if 0 < pp <= c['pilot_peak'] and 0 < pd <= c['data_peak']:
                result.append((tau, pp, pd))
    if not result:
        raise ValueError('No feasible action under energy and peak-power limits')
    return np.asarray(result)


def sample(c, speed, load, rng):
    speeds = np.maximum(0, rng.normal(speed, c['speed_std'], c['samples']))
    doppler = speeds / 3.6 * c['carrier_hz'] / 299792458
    rho = j0(2 * np.pi * doppler * c['separation_seconds'])
    users = np.maximum(1, rng.poisson(max(load - 1, 0), c['samples']) + 1)
    return rho, users


def bler(a, snr, rho, load, c):
    tau, pp, pd = a
    nd = c['blocklength'] - tau
    quality = tau * pp / (1 + tau * pp)
    gamma = 10 ** (snr / 10)
    sinr = gamma * pd * quality * rho ** 2 / (
        1 + gamma * pd * (1 - rho ** 2) + c['interference'] * (load - 1))
    dispersion = (1 - (1 + sinr) ** -2) * np.log2(np.e) ** 2
    z = (np.log2(1 + sinr) - c['bits'] / nd) / np.sqrt(np.maximum(dispersion, 1e-15) / nd)
    return ndtr(-z)


def prepare(c, out):
    rng = np.random.default_rng(c['seed'])
    records = {}
    for i, speed in enumerate(c['speeds']):
        for j, load in enumerate(c['loads']):
            rho, users = sample(c, speed, load, rng)
            records[f'rho_{i}_{j}'] = rho
            records[f'load_{i}_{j}'] = users
    np.savez_compressed(out / 'design.npz', **records)


def fit(c, out):
    design = np.load(out / 'design.npz')
    grid = actions(c)
    rows = []
    for snr in c['snr_db']:
        for i, speed in enumerate(c['speeds']):
            for j, load in enumerate(c['loads']):
                rho, users = design[f'rho_{i}_{j}'], design[f'load_{i}_{j}']
                errors = np.stack([bler(a, snr, rho, users, c) for a in grid])
                risk = errors.mean(1) + c['risk_weight'] * np.array([tail_mean(e, c['alpha']) for e in errors])
                blind = np.stack([bler(a, snr, np.ones_like(rho), users, c) for a in grid])
                fixed = np.argmin(abs(grid[:, 0] - c['fixed_pilot']) +
                                  abs(grid[:, 0] * grid[:, 1] / c['energy'] - c['fixed_fraction']))
                for name, idx in [('DopplerPilot', risk.argmin()), ('mean-only', errors.mean(1).argmin()),
                                  ('aging-blind', blind.mean(1).argmin()), ('fixed', fixed)]:
                    tau, pp, data_power = grid[idx]
                    rows.append(dict(method=name, snr=snr, speed=speed, load=load, tau=int(tau), pp=pp, pd=data_power))
    pd.DataFrame(rows).to_csv(out / 'policy.csv', index=False)


def run(c, out):
    policy = pd.read_csv(out / 'policy.csv')
    rng = np.random.default_rng(c['seed'] + 10000)
    rows = []
    for state, choices in policy.groupby(['snr', 'speed', 'load'], sort=False):
        snr, speed, load = state
        rho, users = sample(c, speed, load, rng)
        grid = actions(c)
        oracle = np.stack([bler(a, snr, rho, users, c) for a in grid])
        best = oracle.argmin(0)
        for choice in choices.itertuples():
            start = time.perf_counter_ns()
            selected = policy[(policy.snr == snr) & (policy.speed == speed) & (policy.load == load) &
                              (policy.method == choice.method)].iloc[0]
            lookup_us = (time.perf_counter_ns() - start) / 1000
            a = np.array([selected.tau, selected.pp, selected.pd])
            e = bler(a, snr, rho, users, c)
            idx = np.argmin(np.linalg.norm(grid - a, axis=1))
            for k in range(len(e)):
                rows.append(dict(method=choice.method, snr=snr, speed=speed, load=load,
                                 sample=k, bler=e[k], oracle_bler=oracle[best[k], k],
                                 oracle_match=int(idx == best[k]), lookup_us=lookup_us,
                                 goodput=c['bits'] * (1 - e[k]) / c['blocklength']))
    pd.DataFrame(rows).to_csv(out / 'packets.csv', index=False)


def evaluate(c, out):
    data = pd.read_csv(out / 'packets.csv')
    rows = []
    for key, g in data.groupby(['method', 'snr', 'speed', 'load']):
        rows.append(dict(zip(['method', 'snr', 'speed', 'load'], key), mean_bler=g.bler.mean(),
                         cvar_bler=tail_mean(g.bler.to_numpy(), c['alpha']),
                         violation=float((g.bler > c['reliability_threshold']).mean()),
                         goodput=g.goodput.mean(), decision_accuracy=g.oracle_match.mean(),
                         online_lookup_us=g.lookup_us.mean()))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    data = pd.read_csv(out / 'metrics.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in data.groupby('method'):
        curve = g.groupby('snr').mean(numeric_only=True)
        ax.semilogy(curve.index, np.maximum(curve.mean_bler, 1e-15), marker='o', label=name)
    ax.set(xlabel='SNR (dB)', ylabel='Mean surrogate BLER')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'bler.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'fit', 'run', 'evaluate', 'plot'])
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not 0 < c['alpha'] < 1:
        raise ValueError('alpha must lie in (0,1)')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
