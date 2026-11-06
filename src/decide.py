"""Pre-registered decision rules for the 2026-11-06 day queue. Prints ONE token to stdout; reasons go to stderr and runs/decision_<mode>.json.
  python src/decide.py T2        -> prints 150 if T2 passes else 300
  python src/decide.py T1 <EP>   -> prints ADOPT or REJECT (T1 vs its same-epoch reference)
Best fold-0 val MAE is read from each run's fold0_log.csv; a run without fold0_test.npy counts as missing."""
import sys, os, json
import numpy as np, pandas as pd
ROOT = os.path.expanduser('~/ventilator-pressure-prediction')
def best(r):
    if not os.path.exists(f'{ROOT}/runs/{r}/fold0_test.npy'): raise FileNotFoundError(r)
    return float(pd.read_csv(f'{ROOT}/runs/{r}/fold0_log.csv').val_mae.min())
CTRL = ['exp06', 'exp08', 'exp09']          # exp06 config, 300 epochs, seeds 42/43/44
mode = sys.argv[1]; out = {'mode': mode}
try:
    c = np.array([best(r) for r in CTRL]); s_c = c.std(ddof=1); out.update(ctrl=dict(zip(CTRL, c.round(5).tolist())), ctrl_mean=c.mean(), s_ctrl=s_c)
    if mode == 'T2':
        t = np.array([best(r) for r in ('exp10a', 'exp10b')])
        thr = c.mean() + 2 * s_c * np.sqrt(1/2 + 1/3)
        ok = t.mean() <= thr
        out.update(t2=t.round(5).tolist(), t2_mean=t.mean(), threshold=thr, diff=t.mean() - c.mean(), passed=bool(ok)); token = '150' if ok else '300'
    else:
        ep = int(sys.argv[2]); t = np.array([best(r) for r in ('exp11a', 'exp11b')])
        if ep == 150:
            ref = np.array([best(r) for r in ('exp10a', 'exp10b')]); s = np.sqrt((2 * s_c**2 + 1 * ref.std(ddof=1)**2) / 3); se = s * np.sqrt(1/2 + 1/2)
        else:
            ref = c; s = s_c; se = s * np.sqrt(1/2 + 1/3)
        thr = ref.mean() - 2 * se; ok = t.mean() < thr
        out.update(epochs=ep, t1=t.round(5).tolist(), t1_mean=t.mean(), ref=ref.round(5).tolist(), ref_mean=ref.mean(), s_pooled=s, threshold=thr, diff=t.mean() - ref.mean(), adopted=bool(ok))
        token = 'ADOPT' if ok else 'REJECT'
except FileNotFoundError as e:
    out['missing'] = str(e); token = '300' if mode == 'T2' else 'REJECT'
out = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}
json.dump(out, open(f'{ROOT}/runs/decision_{mode}.json', 'w'), indent=1, default=float)
sys.stderr.write(f'decide {mode}: {out} -> {token}\n'); print(token)
