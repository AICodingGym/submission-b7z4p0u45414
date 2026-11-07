"""Pre-registered 2-seed (or n-seed) arm vs the N1 reference (exp12a/b/c, bs 128), fold 0.
Rule: ADOPT if mean(arm) < mean(ref) - 2*s*sqrt(1/n + 1/3), s = pooled SD of the old controls (exp06/08/09) and N1 = 0.00453.
Usage: python src/decide_vs_ref.py NAME run1 run2 [...]  -> prints ADOPT/REJECT/INCOMPLETE, writes runs/decision_NAME.json"""
import sys, json, os, numpy as np, pandas as pd
name, runs = sys.argv[1], sys.argv[2:]
REF = ['exp12a', 'exp12b', 'exp12c']; CTRL = ['exp06', 'exp08', 'exp09']
def best(r):
    d = f'runs/{r}'
    if not os.path.exists(f'{d}/fold0_test.npy'): return None    # fold not finished
    return float(pd.read_csv(f'{d}/fold0_log.csv').val_mae.min())
ref = [best(r) for r in REF]; ctrl = [best(r) for r in CTRL]; arm = {r: best(r) for r in runs}
s = float(np.sqrt((np.var(ctrl, ddof=1) + np.var(ref, ddof=1)) / 2))
done = [v for v in arm.values() if v is not None]; n = len(done)
out = dict(name=name, ref=dict(zip(REF, ref)), ref_mean=round(float(np.mean(ref)), 5), s_pooled=round(s, 5), arm=arm)
if n < len(runs): out.update(verdict='INCOMPLETE', adopted=False)
else:
    thr = float(np.mean(ref) - 2 * s * np.sqrt(1 / n + 1 / 3)); m = float(np.mean(done))
    out.update(n=n, mean=round(m, 5), diff=round(m - float(np.mean(ref)), 5), threshold=round(thr, 5), adopted=bool(m < thr), verdict='ADOPT' if m < thr else 'REJECT')
json.dump(out, open(f'runs/decision_{name}.json', 'w'), indent=1); print(out['verdict'])
