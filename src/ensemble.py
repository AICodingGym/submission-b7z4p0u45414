"""Average per-fold test predictions of one or more runs into submission.csv and report OOF MAE.
Usage: python src/ensemble.py exp02 [exp03 ...] [--snap]
"""
import sys, glob, os, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/ventilator-pressure-prediction')
runs = [a for a in sys.argv[1:] if not a.startswith('--')]; snap = '--snap' in sys.argv
files = sorted(f for r in runs for f in glob.glob(f'{ROOT}/runs/{r}/fold*_test.npy'))
print('using', [os.path.relpath(f, ROOT) for f in files])
P = np.stack([np.load(f) for f in files])
pred = np.median(P, 0) if len(files) > 2 else P.mean(0)
if snap:
    grid = np.unique(pd.read_csv(f'{ROOT}/data/train.csv', usecols=['pressure']).pressure.values)
    i = np.clip(np.searchsorted(grid, pred), 1, len(grid) - 1)
    pred = np.where(np.abs(grid[i] - pred) < np.abs(grid[i - 1] - pred), grid[i], grid[i - 1])
te = pd.read_csv(f'{ROOT}/data/test.csv', usecols=['id'])
sub = pd.DataFrame({'id': te.id, 'pressure': pred})
assert len(sub) == 603600 and sub.pressure.notna().all()
sub.to_csv(f'{ROOT}/submission.csv', index=False); print('wrote submission.csv', sub.shape, sub.pressure.describe().round(3).to_dict())
