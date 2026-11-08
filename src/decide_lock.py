"""Sunday-evening config LOCK (pre-registered in EXPERIMENT_LOG.md "Sunday plan"; committed BEFORE exp16c's result exists).
Rule: an arm is adopted if mean over ALL its seeds (fold 0) < ref_mean - 2*s*sqrt(1/n + 1/3); ref = N1 exp12a/b/c (bs128/h128);
s = pooled SD of the old controls (exp06/08/09) and N1 (0.00453). A third seed can overturn a 2-seed pass.
If both arms are adopted, the untested combination is not stacked: pick the lower E(K) = s1 - d*(1 - 1/K), with K = runs in 45 GPU-h
and d = 2*(mean single - mean pair-average) from the arm's own seeds. If none is adopted: lock bs128/h128.
Refuses (exit 1) if a planned run is not finished.  Usage: python src/decide_lock.py  -> prints LOCK <config>, writes runs/decision_LOCK.json"""
import json, os, sys, itertools, numpy as np, pandas as pd
REF = ['exp12a', 'exp12b', 'exp12c']; CTRL = ['exp06', 'exp08', 'exp09']
ARMS = {'bs64/h128': dict(runs=['exp16a', 'exp16b', 'exp16c'], min_per_run=100),     # D1 (seed 44 added Sunday)
        'h256/bs128': dict(runs=['exp18a', 'exp18b'], min_per_run=205)}             # H1 (not close: no third seed)
def best(r):
    if not os.path.exists(f'runs/{r}/fold0_test.npy'): return None
    return float(pd.read_csv(f'runs/{r}/fold0_log.csv').val_mae.min())
ref = [best(r) for r in REF]; ctrl = [best(r) for r in CTRL]
s = float(np.sqrt((np.var(ctrl, ddof=1) + np.var(ref, ddof=1)) / 2)); ref_mean = float(np.mean(ref))
out = dict(ref=dict(zip(REF, ref)), ref_mean=round(ref_mean, 5), s_pooled=round(s, 5), arms={})
missing = [r for a in ARMS.values() for r in a['runs'] if best(r) is None]
if missing: print('INCOMPLETE', missing); sys.exit(1)
for name, a in ARMS.items():
    v = [best(r) for r in a['runs']]; n = len(v); m = float(np.mean(v)); thr = ref_mean - 2 * s * np.sqrt(1 / n + 1 / 3)
    out['arms'][name] = dict(runs=dict(zip(a['runs'], v)), n=n, mean=round(m, 5), diff=round(m - ref_mean, 5), threshold=round(float(thr), 5), adopted=bool(m < thr))
adopted = [k for k, a in out['arms'].items() if a['adopted']]
if len(adopted) == 0: lock = 'bs128/h128'
elif len(adopted) == 1: lock = adopted[0]
else:
    sys.path.insert(0, 'src'); from common import load_train, make_folds, targets
    tr = load_train(); _, fold_of = make_folds(tr); Y, M = targets(tr); v0 = fold_of == 0; Yv, Mv = Y[v0], M[v0].astype(bool)
    mae = lambda P: float(np.abs(P - Yv)[Mv].mean()); E = {}
    for k in adopted:
        O = [np.load(f'runs/{r}/fold0_oof.npy') for r in ARMS[k]['runs']]
        s1 = np.mean([mae(o) for o in O]); s2 = np.mean([mae((a + b) / 2) for a, b in itertools.combinations(O, 2)])
        d = 2 * (s1 - s2); K = int(45 * 60 // ARMS[k]['min_per_run']); E[k] = s1 - d * (1 - 1 / K)
        out['arms'][k].update(d=round(float(d), 5), K=K, E=round(float(E[k]), 5))
    lock = min(E, key=E.get)
out['lock'] = lock; json.dump(out, open('runs/decision_LOCK.json', 'w'), indent=1); print('LOCK', lock)
