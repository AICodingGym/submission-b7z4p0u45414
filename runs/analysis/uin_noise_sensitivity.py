"""How much does exp06 (fold-0 model) change its predicted pressure when u_in gets iid Gaussian noise? Proxy for how far a noisy-u_in
training example's label would be from its true pressure (the model stands in for the unknown physics). CPU."""
import sys, os, ast, json, types, time
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, 'src'); from common import load_train, make_folds, targets
torch.set_num_threads(4)
tr = load_train(); bid, fold_of = make_folds(tr)
va_ids = bid[fold_of == 0]; sub = tr[tr.breath_id.isin(va_ids[:4000])].copy()   # 4,000 fold-0 val breaths
code = open('src/train_lstm_v2.py').read(); tree = ast.parse(code)
defs = {n.name: ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
g = dict(np=np, pd=pd, torch=torch, nn=nn); exec(defs['feats'], g)
g['args'] = types.SimpleNamespace(**json.load(open('runs/exp06/fold0_args.json'))); exec(defs['Net'], g)
mu, sd = np.load('runs/exp06/fold0_scaler.npy'); st = {}
def predict(df):
    f = g['feats'](df)
    if 'net' not in st:
        st['F'] = [c for c in f.columns if c not in ('id', 'breath_id', 'pressure')]
        st['net'] = g['Net'](len(st['F'])); st['net'].load_state_dict(torch.load('runs/exp06/fold0_best.pt', map_location='cpu')); st['net'].eval()
    X = ((f[st['F']].values.astype('float32').reshape(-1, 80, len(st['F']))) - mu) / sd
    with torch.no_grad(): return np.concatenate([st['net'](torch.from_numpy(X[i:i + 1000])).numpy() for i in range(0, len(X), 1000)])
m = (sub.u_out.values == 0).reshape(-1, 80); y = sub.pressure.values.reshape(-1, 80); u = sub.u_in.values.reshape(-1, 80)
r50 = sub.R.values.reshape(-1, 80)[:, 0] == 50
P0 = predict(sub); base = np.abs(P0 - y)[m].mean()
print(f'4000 val breaths: model MAE {base:.4f}; u_in on scored steps: median {np.median(u[m]):.2f}, mean {u[m].mean():.2f}, p10 {np.percentile(u[m],10):.2f}, p90 {np.percentile(u[m],90):.2f}; pressure grid step 0.0703', flush=True)
rng = np.random.RandomState(0); rows = []
for s in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
    d = sub.copy(); d['u_in'] = np.clip(d.u_in.values + rng.normal(0, s, len(d)), 0, 100)
    P = predict(d); dP = np.abs(P - P0)
    rows.append(dict(sigma=s, mean_abs_dP=dP[m].mean(), p95_abs_dP=np.percentile(dP[m], 95), mean_abs_dP_R50=dP[r50][m[r50]].mean(), mae_vs_truth=np.abs(P - y)[m].mean()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(5).to_csv('runs/analysis/uin_noise_sensitivity.csv', index=False); print('DONE')
