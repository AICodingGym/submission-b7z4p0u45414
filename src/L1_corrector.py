"""L1 (pre-registered 2026-11-06): CPU level-corrector on exp06 OOF residuals.
For each fold k: train HistGradientBoosting on the OTHER 4 folds' breaths (target = per-breath mean signed error of exp06's OOF on u_out==0 steps),
predict the offset for fold k's breaths, subtract it, and re-score. Model unchanged; no test data used."""
import sys, os, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, 'src'); from common import load_train, make_folds, targets
t0 = time.time()
tr = load_train(); bid, fold_of = make_folds(tr); Y, M = targets(tr)
U = tr.u_in.values.reshape(-1, 80); UO = tr.u_out.values.reshape(-1, 80); T = tr.time_step.values.reshape(-1, 80)
R = tr.R.values.reshape(-1, 80)[:, 0]; C = tr.C.values.reshape(-1, 80)[:, 0]
P = np.zeros_like(Y)
for k in range(5): P[fold_of == k] = np.load(f'runs/exp06/fold{k}_oof.npy')
off = ((P - Y) * M).sum(1) / M.sum(1)
dt = np.diff(T, axis=1); n_insp = M.sum(1)
Pins = np.where(M, P, np.nan)
F = np.column_stack([U, P[:, :32], R, C, n_insp, dt.mean(1), dt.max(1), T[:, -1], np.nanmean(Pins, 1), np.nanmax(Pins, 1), UO.sum(1)])
print('features', F.shape, 'prep', round(time.time() - t0), 's', flush=True)
def score(P_, k): s = fold_of == k; return np.abs(P_[s] - Y[s])[M[s]].mean()
rows = []; pred_off = {}
for loss in ('squared_error', 'absolute_error'):
    po = np.zeros(len(Y))
    for k in range(5):
        tk, vk = fold_of != k, fold_of == k
        m = HistGradientBoostingRegressor(loss=loss, max_iter=500, learning_rate=0.05, max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1, random_state=0)
        m.fit(F[tk], off[tk]); po[vk] = m.predict(F[vk])
        Pc = P - po[:, None]
        rows.append(dict(loss=loss, fold=k, base=score(P, k), corrected=score(Pc, k), gain=score(P, k) - score(Pc, k),
                         r2=1 - ((off[vk] - po[vk])**2).sum() / ((off[vk] - off[vk].mean())**2).sum(), iters=m.n_iter_))
        print(rows[-1], flush=True)
    pred_off[loss] = po
df = pd.DataFrame(rows); df.round(5).to_csv('runs/analysis/L1_corrector.csv', index=False)
for loss, d in df.groupby('loss', sort=False):
    Pc = P - pred_off[loss][:, None]
    pooled_b = np.abs(P - Y)[M].mean(); pooled_c = np.abs(Pc - Y)[M].mean()
    r2 = 1 - ((off - pred_off[loss])**2).sum() / ((off - off.mean())**2).sum()
    print(f'{loss}: pooled OOF {pooled_b:.5f} -> {pooled_c:.5f} (gain {pooled_b - pooled_c:.5f}); pooled CV R2 {r2:.4f}; folds improved >= 0.002: {(d.gain >= 0.002).sum()}/5; mean fold gain {d.gain.mean():.5f}', flush=True)
print('L1 DONE', round(time.time() - t0), 's', flush=True)
