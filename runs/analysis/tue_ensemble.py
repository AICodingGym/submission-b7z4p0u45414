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
C6 = ['exp04','exp06','exp07','exp08','exp09','exp12a','exp12b','exp12c','exp16a','exp16b','exp12d','exp16c']; C7 = C6 + ['exp12e']
for name, s in [('C6 (sub11 members)', C6), ('C7 (C6+exp12e)', C7)]:
    print('F0', name, len(s), 'mean, mean+snap', score(s, 0))
def full(runs, do_snap=False):
    e = []
    for k in range(5):
        v = fold_of == k; P = np.mean([np.load(f'runs/{r}/fold{k}_oof.npy') for r in runs], 0)
        if do_snap: P = snap(P)
        e.append(np.abs(P - Y[v])[M[v].astype(bool)])
    return round(float(np.concatenate(e).mean()), 5)
seeds = ['exp12a','exp12b','exp12c','exp12d','exp12e']
for n in range(1, 6):
    print('FULL5 seeds', n, '+'.join(seeds[:n]), 'mean', full(seeds[:n]), 'mean+snap', full(seeds[:n], True))
print('FULL5 exp12d alone', full(['exp12d']), '| exp12e alone', full(['exp12e']))
print('FULL5 exp06 + 5 seeds', full(['exp06'] + seeds), 'snap', full(['exp06'] + seeds, True))
