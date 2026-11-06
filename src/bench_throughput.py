"""Throughput test (PA3 3a): time/epoch and GPU utilization vs batch size for the v2 model.
Training step is identical to train_lstm_v2.py (AdamW, OneCycle, masked L1, grad clip); LR scaled with batch.
Writes runs/bench_throughput.csv. Usage: python src/bench_throughput.py [--epochs 2]"""
import sys, os, ast, time, json, types, threading, subprocess, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_train, make_folds
ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--layers', type=int, default=4)
cli = ap.parse_args()
code = open(os.path.join(ROOT, 'src/train_lstm_v2.py')).read(); tree = ast.parse(code)
defs = {n.name: ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
g = dict(np=np, pd=pd, torch=torch, nn=nn); exec(defs['feats'], g)
t0 = time.time(); tr = g['feats'](load_train())
FEATS = [c for c in tr.columns if c not in ('id', 'breath_id', 'pressure')]
X = tr[FEATS].values.astype('float32').reshape(-1, 80, len(FEATS)); Y = tr.pressure.values.astype('float32').reshape(-1, 80); M = (tr.u_out.values == 0).reshape(-1, 80)
bid, fold_of = make_folds(tr, 5); trn = fold_of != 0
mu, sd = np.load(f'{ROOT}/runs/exp06/fold0_scaler.npy'); X = (X - mu) / sd
print('prep', round(time.time() - t0), 's', flush=True)
dev = 'cuda'; util = []; stop = [False]
def sampler():
    while not stop[0]:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        try: util.append(float(r.stdout.strip()))
        except ValueError: pass
        time.sleep(0.5)
rows = []
for gpu_data in (False, True):
    for bs in (128, 256, 512):
        lr = 2e-3 * bs / 512
        torch.manual_seed(0)
        g['args'] = types.SimpleNamespace(hidden=cli.hidden, layers=cli.layers, dropout=0.0); exec(defs['Net'], g)
        model = g['Net'](len(FEATS)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        nb = int(np.ceil(trn.sum() / bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=300 * nb, pct_start=0.05)
        Xt, Yt, Mt = [torch.from_numpy(a[trn]) for a in (X, Y, M)]
        if gpu_data: Xt, Yt, Mt = Xt.to(dev), Yt.to(dev), Mt.to(dev)
        for ep in range(cli.epochs):
            util.clear(); stop[0] = False; th = threading.Thread(target=sampler); th.start()
            torch.cuda.synchronize(); t1 = time.time(); model.train()
            perm = torch.randperm(len(Xt), device=Xt.device)
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]; x, y, m = Xt[b].to(dev), Yt[b].to(dev), Mt[b].to(dev)
                err = (model(x) - y).abs(); loss = err[m].mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            torch.cuda.synchronize(); ttrain = time.time() - t1
            stop[0] = True; th.join()
            row = dict(gpu_data=gpu_data, bs=bs, lr=lr, epoch=ep, steps=nb, train_sec=round(ttrain, 2), ms_per_step=round(1000 * ttrain / nb, 1),
                       gpu_util_mean=round(float(np.mean(util)), 1) if util else None, gpu_util_median=float(np.median(util)) if util else None, n_util=len(util),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            rows.append(row); pd.DataFrame(rows).to_csv(f'{ROOT}/runs/bench_throughput.csv', index=False); print(row, flush=True)
        del model, opt, Xt, Yt, Mt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print('BENCH DONE', flush=True)
