import sys, os, ast, json, types
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, 'src'); from common import ROOT, load_train, make_folds, targets
torch.set_num_threads(4)
tr = load_train(); bid, fold_of = make_folds(tr); Y, M = targets(tr); tn = fold_of != 0
code = open('src/train_lstm_v2.py').read(); tree = ast.parse(code)
defs = {n.name: ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
g = dict(np=np, pd=pd, torch=torch, nn=nn); exec(defs['feats'], g); ft = g['feats'](tr)
FEATS = [c for c in ft.columns if c not in ('id', 'breath_id', 'pressure')]
X = ft[FEATS].values.astype('float32').reshape(-1, 80, len(FEATS)); del ft
g['args'] = types.SimpleNamespace(**json.load(open('runs/exp06/fold0_args.json'))); exec(defs['Net'], g)
net = g['Net'](len(FEATS)); net.load_state_dict(torch.load('runs/exp06/fold0_best.pt', map_location='cpu')); net.eval()
mu, sd = np.load('runs/exp06/fold0_scaler.npy'); Xt = (X[tn] - mu) / sd
with torch.no_grad(): P = np.concatenate([net(torch.from_numpy(Xt[i:i+1024])).numpy() for i in range(0, len(Xt), 1024)])
np.save('runs/analysis/exp06_fold0_trainpred.npy', P)
Yt, Mt = Y[tn], M[tn]; R = tr.R.values.reshape(-1, 80)[tn, 0]
err = P - Yt; off = (err * Mt).sum(1) / Mt.sum(1); tot = np.abs(err)[Mt].mean(); shape = np.abs(err - off[:, None])[Mt].mean()
print('TRAIN fold (exp06 f0, eval mode): MAE', round(tot, 5), 'shape-only (oracle)', round(shape, 5), 'level share', round(1 - shape / tot, 3),
      '| |offset| median', round(np.median(np.abs(off)), 3), 'p95', round(np.percentile(np.abs(off), 95), 3), 'mean', round(np.abs(off).mean(), 3))
for r in (5, 20, 50):
    s = R == r; print(' R', r, 'mean |offset| train', round(np.abs(off[s]).mean(), 3), 'level share', round(1 - np.abs(err - off[:, None])[s][Mt[s]].mean() / np.abs(err)[s][Mt[s]].mean(), 3))
