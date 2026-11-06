"""Eval-mode MAE of a v2 run's best checkpoint on its train fold and val fold (u_out==0 steps only). CPU OK.
Usage: python src/eval_train.py exp06:0 exp07:0 ...   -> appends to runs/eval_train.csv
Features and Net are taken verbatim (via ast) from the script whose md5 the fold's args file records."""
import sys, os, ast, json, time, hashlib, types
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_train, make_folds
torch.set_num_threads(8)
SRC = os.path.join(ROOT, 'src/train_lstm_v2.py'); code = open(SRC).read()
MD5 = hashlib.md5(code.encode()).hexdigest()
tree = ast.parse(code)
defs = {n.name: ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
g = dict(np=np, pd=pd, torch=torch, nn=nn); exec(defs['feats'], g)
tr = g['feats'](load_train())
FEATS = [c for c in tr.columns if c not in ('id', 'breath_id', 'pressure')]
X_raw = tr[FEATS].values.astype('float32').reshape(-1, 80, len(FEATS))
Y = tr.pressure.values.astype('float32').reshape(-1, 80); M = (tr.u_out.values == 0).reshape(-1, 80)
bid, fold_of = make_folds(tr, 5)
out = os.path.join(ROOT, 'runs/eval_train.csv')
for spec in sys.argv[1:]:
    run, k = spec.split(':'); k = int(k); d = f'{ROOT}/runs/{run}'
    a = json.load(open(f'{d}/fold{k}_args.json')); assert a['script_md5'] == MD5, (spec, a['script_md5'], MD5)
    g['args'] = types.SimpleNamespace(**a); exec(defs['Net'], g)
    model = g['Net'](len(FEATS)); model.load_state_dict(torch.load(f'{d}/fold{k}_best.pt', map_location='cpu')); model.eval()
    mu, sd = np.load(f'{d}/fold{k}_scaler.npy'); X = (X_raw - mu) / sd
    t0 = time.time(); P = []
    with torch.no_grad():
        for i in range(0, len(X), 1024): P.append(model(torch.from_numpy(X[i:i+1024])).numpy())
    P = np.concatenate(P); va = fold_of == k
    e = np.abs(P - Y)
    row = dict(run=run, fold=k, train_mae=e[~va][M[~va]].mean(), val_mae=e[va][M[va]].mean(),
               log_best=pd.read_csv(f'{d}/fold{k}_log.csv').val_mae.min(), oof_file_mae=np.abs(np.load(f'{d}/fold{k}_oof.npy') - Y[va])[M[va]].mean(),
               sec=round(time.time() - t0))
    row['ratio'] = row['val_mae'] / row['train_mae']
    pd.DataFrame([row]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
    print(row, flush=True)
