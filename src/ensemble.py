"""Combine runs: report OOF MAE for MEAN and MEDIAN aggregation, then write submission.csv.

Usage: python src/ensemble.py exp04 exp05 [exp06 ...] [--agg mean|median] [--snap]

OOF comparison: in each fold, every OOF breath is predicted by exactly one model PER RUN.
So mean vs median can only be compared ACROSS RUNS (seeds / variants) on the same fold.
With 2 runs, mean == median (median of 2 values is their mean); you need >= 3 runs for a real test.
Test predictions: all fold x run predictions are combined with --agg (default mean; choose it from the OOF table).
"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_train, make_folds, targets

runs = [a for a in sys.argv[1:] if not a.startswith('--')]
agg = sys.argv[sys.argv.index('--agg') + 1] if '--agg' in sys.argv else 'mean'
snap = '--snap' in sys.argv
assert agg in ('mean', 'median') and runs

tr = load_train(); _, fold_of = make_folds(tr); Y, M = targets(tr)
grid = np.unique(tr.pressure.values)


def snap_to_grid(p):
    i = np.clip(np.searchsorted(grid, p), 1, len(grid) - 1)
    return np.where(np.abs(grid[i] - p) < np.abs(grid[i - 1] - p), grid[i], grid[i - 1])


def mae(P, k):
    return np.abs(P - Y[fold_of == k])[M[fold_of == k]].mean()


# ---- OOF table: per run, then mean / median across runs, per fold ----
rows = []
for k in range(5):
    # only COMPLETED folds: a killed run leaves a partial oof.npy but never writes test.npy
    have = [r for r in runs if os.path.exists(f'{ROOT}/runs/{r}/fold{k}_oof.npy')
            and os.path.exists(f'{ROOT}/runs/{r}/fold{k}_test.npy')]
    if not have:
        continue
    P = np.stack([np.load(f'{ROOT}/runs/{r}/fold{k}_oof.npy') for r in have])
    row = {'fold': k, 'n_runs': len(have)}
    for r, p in zip(have, P):
        row[r] = mae(p, k)
    row['MEAN'] = mae(P.mean(0), k)
    row['MEDIAN'] = mae(np.median(P, 0), k)
    row['MEAN+snap'] = mae(snap_to_grid(P.mean(0)), k)
    row['MEDIAN+snap'] = mae(snap_to_grid(np.median(P, 0)), k)
    rows.append(row)
tab = pd.DataFrame(rows).set_index('fold')
pd.set_option('display.width', 200)
print('OOF MAE on u_out==0 (lower is better):'); print(tab.round(5))
if len(tab) and (tab.n_runs < 3).any():
    print('NOTE: folds with < 3 runs cannot distinguish mean from median.')

# ---- test submission ----
files = sorted(f for r in runs for f in glob.glob(f'{ROOT}/runs/{r}/fold*_test.npy'))
print('test files used:', [os.path.relpath(f, ROOT) for f in files], '| agg =', agg, '| snap =', snap)
T = np.stack([np.load(f) for f in files])
pred = T.mean(0) if agg == 'mean' else np.median(T, 0)
if snap:
    pred = snap_to_grid(pred)
ss = pd.read_csv(f'{ROOT}/data/sample_submission.csv')
sub = pd.DataFrame({'id': ss.id.values, 'pressure': pred})
assert len(sub) == 603600 and np.isfinite(sub.pressure).all()
sub.to_csv(f'{ROOT}/submission.csv', index=False)
print('wrote submission.csv', sub.shape, '(NOT submitted)')
