import ast, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, 'src'); from common import load_train
code = open('src/train_lstm_v3.py').read(); tree = ast.parse(code)
defs = {n.name: ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
g = dict(np=np, pd=pd, torch=torch); exec(defs['feats'], g)
tr = load_train(); tr = tr[tr.breath_id.isin(tr.breath_id.unique()[:500])]
f0 = g['feats'](tr); FEATS = [c for c in f0.columns if c not in ('id', 'breath_id', 'pressure')]
g['FEATS'] = FEATS; g['IDX'] = {c: i for i, c in enumerate(FEATS)}; exec(defs['recompute_uin_feats'], g)
xr = torch.from_numpy(f0[FEATS].values.astype('float32').reshape(-1, 80, len(FEATS)))
sd = f0[FEATS].values.std(0) + 1e-6
# (1) no noise: recompute must reproduce feats()
a = g['recompute_uin_feats'](xr, xr[..., g['IDX']['u_in']].clone()).numpy()
print('no-noise max |diff| / sd:', float((np.abs(a - xr.numpy()) / sd).max()))
# (2) with noise: recompute vs feats() on a dataframe whose u_in got the same noise
torch.manual_seed(0); u = xr[..., g['IDX']['u_in']]; un = (u * (1 + 0.01 * torch.randn_like(u))).clamp(0, 100)
b = g['recompute_uin_feats'](xr, un).numpy()
d = tr.copy(); d['u_in'] = un.numpy().reshape(-1).astype('float64'); f1 = g['feats'](d)[FEATS].values.reshape(-1, 80, len(FEATS))
err = np.abs(b - f1) / sd; print('noisy path max |diff| / sd:', float(err.max()), 'worst column', FEATS[int(err.max(axis=(0, 1)).argmax())])
print('zeros stay zero:', bool((un[u == 0] == 0).all()), '| mean relative change on u>0:', float(((un - u).abs() / u)[u > 0].mean()))
