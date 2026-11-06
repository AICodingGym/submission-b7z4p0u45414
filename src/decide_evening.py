"""Pre-registered evening verdicts (2026-11-06). Controls: exp06/exp08/exp09 (300 ep, seeds 42/43/44).
ADOPT an arm if mean(arm) < mean(controls) - 2 * s * sqrt(1/n_arm + 1/3), s = SD (ddof 1) of the 3 controls. Missing runs are skipped (n shrinks)."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/ventilator-pressure-prediction')
def best(r):
    return float(pd.read_csv(f'{ROOT}/runs/{r}/fold0_log.csv').val_mae.min()) if os.path.exists(f'{ROOT}/runs/{r}/fold0_test.npy') else None
c = np.array([best(r) for r in ('exp06', 'exp08', 'exp09')]); s = c.std(ddof=1)
arms = {'N1 bs128': ['exp12a', 'exp12b', 'exp12c'], 'N2 wd1e-2': ['exp07', 'exp13b', 'exp13c'], 'N4 uin_noise 0.01': ['exp15a', 'exp15b'], 'N5 T1 seed 44 (with exp11a/b)': ['exp11a', 'exp11b', 'exp11c']}
out = {}
for k, runs in arms.items():
    v = {r: best(r) for r in runs}; got = np.array([x for x in v.values() if x is not None])
    if len(got) == 0: out[k] = 'no finished runs'; continue
    thr = c.mean() - 2 * s * np.sqrt(1 / len(got) + 1 / 3)
    out[k] = dict(runs=v, n=len(got), mean=round(got.mean(), 5), ctrl_mean=round(c.mean(), 5), diff=round(got.mean() - c.mean(), 5), threshold=round(thr, 5), adopt=bool(got.mean() < thr))
json.dump(out, open(f'{ROOT}/runs/decision_evening.json', 'w'), indent=1); print(json.dumps(out, indent=1))
