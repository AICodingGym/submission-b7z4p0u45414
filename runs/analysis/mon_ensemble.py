"""Sunday ensemble check (pre-registered in the log). CPU, single thread."""
import os; os.environ['OMP_NUM_THREADS'] = '1'
import sys, numpy as np
sys.path.insert(0, 'src'); from common import load_train, make_folds, targets
tr = load_train(); _, fold_of = make_folds(tr); Y, M = targets(tr); grid = np.unique(tr.pressure.values); del tr
def snap(p):
    i = np.clip(np.searchsorted(grid, p), 1, len(grid) - 1)
    return np.where(np.abs(grid[i] - p) < np.abs(grid[i - 1] - p), grid[i], grid[i - 1])
def score(runs, k):
    v = fold_of == k; P = np.mean([np.load(f'runs/{r}/fold{k}_oof.npy') for r in runs], 0)
    return round(float(np.abs(P - Y[v])[M[v].astype(bool)].mean()), 5), round(float(np.abs(snap(P) - Y[v])[M[v].astype(bool)].mean()), 5)
C4 = ['exp04','exp06','exp07','exp08','exp09','exp12a','exp12b','exp12c','exp16a','exp16b']; C6 = C4 + ['exp12d','exp16c']
for name, s in [('C4 (sub10 members)', C4), ('C6 (C4+exp12d+exp16c)', C6)]:
    print('F0', name, len(s), 'mean, mean+snap', score(s, 0))
def full(runs):
    err = []; 
    for k in range(5):
        v = fold_of == k; P = np.mean([np.load(f'runs/{r}/fold{k}_oof.npy') for r in runs], 0); m = M[v].astype(bool)
        err.append(np.abs(P - Y[v])[m]); err.append(None) if False else None
    e = np.concatenate([x for x in err if x is not None]); return round(float(e.mean()), 5)
for s in (['exp12a'], ['exp12b'], ['exp12c'], ['exp12a','exp12b'], ['exp12a','exp12b','exp12c']):
    print('FULL5', '+'.join(s), full(s))
