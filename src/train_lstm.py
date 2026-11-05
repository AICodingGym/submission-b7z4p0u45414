"""BiLSTM sequence model for ventilator pressure prediction (PA3, DSC 102).
Usage: python src/train_lstm.py --name exp01 --folds 0 --epochs 200
Writes: runs/<name>/fold<k>_{best.pt,oof.npy,test.npy,log.csv}
"""
import argparse, os, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--folds', default='0')          # e.g. "0" or "0,1,2,3,4"
p.add_argument('--nfolds', type=int, default=5)
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--hidden', type=int, default=256)
p.add_argument('--layers', type=int, default=4)
p.add_argument('--loss_mask', default='insp')   # 'insp' = only u_out==0 steps, 'all' = every step
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

ROOT = os.path.expanduser('~/ventilator-pressure-prediction')
OUT = os.path.join(ROOT, 'runs', args.name); os.makedirs(OUT, exist_ok=True)
json.dump(vars(args), open(os.path.join(OUT, 'args.json'), 'w'), indent=1)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'

def feats(df):
    df = df.copy()
    g = df.groupby('breath_id')
    df['dt'] = g.time_step.diff().fillna(0)
    df['vol'] = (df.u_in * df.dt).groupby(df.breath_id).cumsum()     # volume proxy
    df['u_in_cum'] = g.u_in.cumsum()
    for k in (1, 2, 3, 4):
        df[f'lag{k}'] = g.u_in.shift(k).fillna(0)
        df[f'lead{k}'] = g.u_in.shift(-k).fillna(0)
    df['d1'] = df.u_in - df.lag1
    df['d2'] = df.lag1 - df.lag2
    df['u_out_lag1'] = g.u_out.shift(1).fillna(0)
    df['u_out_lead1'] = g.u_out.shift(-1).fillna(1)
    df['R_'] = df.R.map({5: 0., 20: 1., 50: 2.}); df['C_'] = df.C.map({10: 0., 20: 1., 50: 2.})
    for r in (5, 20, 50): df[f'R{r}'] = (df.R == r).astype('float32')
    for c in (10, 20, 50): df[f'C{c}'] = (df.C == c).astype('float32')
    df['RC'] = df.R * df.C
    df['vol_over_C'] = df.vol / df.C
    df['uin_x_R'] = df.u_in * df.R
    return df

tr = pd.read_csv(os.path.join(ROOT, 'data/train.csv')).sort_values(['breath_id', 'time_step'])
te = pd.read_csv(os.path.join(ROOT, 'data/test.csv'))
te_order = te.id.values
te = te.sort_values(['breath_id', 'time_step'])
tr, te = feats(tr), feats(te)
FEATS = [c for c in tr.columns if c not in ('id', 'breath_id', 'pressure')]
mu = tr[FEATS].mean(); sd = tr[FEATS].std() + 1e-6
def to_arr(df):
    X = ((df[FEATS] - mu) / sd).values.astype('float32').reshape(-1, 80, len(FEATS))
    M = (df.u_out.values == 0).reshape(-1, 80)
    return X, M
Xtr, Mtr = to_arr(tr); Ytr = tr.pressure.values.astype('float32').reshape(-1, 80)
Xte, Mte = to_arr(te)
bid_tr = tr.breath_id.values.reshape(-1, 80)[:, 0]
print('features', len(FEATS), FEATS, flush=True)

class Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, args.hidden), nn.SiLU())
        self.lstm = nn.LSTM(args.hidden, args.hidden, num_layers=args.layers, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.head = nn.Sequential(nn.Linear(2 * args.hidden, args.hidden), nn.SiLU(), nn.Linear(args.hidden, 1))
    def forward(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.head(h).squeeze(-1)

def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(model(torch.from_numpy(X[i:i + 2048]).to(dev)).cpu().numpy())
    return np.concatenate(out)

# breath-level folds, stratified by (R,C)
rc = tr.groupby('breath_id')[['R', 'C']].first().loc[bid_tr]
fold_of = np.zeros(len(bid_tr), int)
rng = np.random.RandomState(0)
for _, idx in pd.Series(np.arange(len(bid_tr))).groupby((rc.R * 100 + rc.C).values):
    idx = idx.values.copy(); rng.shuffle(idx); fold_of[idx] = np.arange(len(idx)) % args.nfolds

for k in [int(f) for f in args.folds.split(',')]:
    va = fold_of == k; trn = ~va
    model = Net(len(FEATS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * int(np.ceil(trn.sum() / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    Xt, Yt, Mt = [torch.from_numpy(a[trn]) for a in (Xtr, Ytr, Mtr)]
    best, log = 1e9, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), args.bs):
            b = perm[i:i + args.bs]
            x, y, m = Xt[b].to(dev), Yt[b].to(dev), Mt[b].to(dev)
            err = (model(x) - y).abs()
            loss = err[m].mean() if args.loss_mask == 'insp' else err.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item() * len(b)
        pv = predict(model, Xtr[va]); vmae = np.abs(pv - Ytr[va])[Mtr[va]].mean()
        log.append(dict(epoch=ep, train=tl / len(Xt), val_mae=vmae, sec=time.time() - t0)); pd.DataFrame(log).to_csv(f'{OUT}/fold{k}_log.csv', index=False)
        if vmae < best:
            best = vmae; torch.save(model.state_dict(), f'{OUT}/fold{k}_best.pt'); np.save(f'{OUT}/fold{k}_oof.npy', pv)
        print(f'fold {k} ep {ep} train {tl/len(Xt):.4f} val_insp_mae {vmae:.4f} best {best:.4f} {time.time()-t0:.1f}s', flush=True)
    model.load_state_dict(torch.load(f'{OUT}/fold{k}_best.pt'))
    pt = predict(model, Xte)
    s = pd.DataFrame({'id': te.id.values, 'pressure': pt.reshape(-1)}).set_index('id').loc[te_order].reset_index()
    np.save(f'{OUT}/fold{k}_test.npy', s.pressure.values)
    s.to_csv(f'{OUT}/fold{k}_submission.csv', index=False)
    print(f'FOLD {k} DONE best val_insp_mae {best:.5f}', flush=True)
