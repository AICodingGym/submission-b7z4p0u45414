"""CPU error analysis on fold 0 (PA3). Writes runs/analysis/*.csv, *.png and summary.txt.
Models: S = exp06 fold 0 (single); E = mean of exp04+exp06+exp07+exp08 fold-0 OOF (4-model ensemble, unsnapped)."""
import sys, os, ast, json, types, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_train, make_folds, targets
torch.set_num_threads(4)
OUT = f'{ROOT}/runs/analysis'; os.makedirs(OUT, exist_ok=True)
log = open(f'{OUT}/summary.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
tr = load_train(); bid, fold_of = make_folds(tr, 5); Y, M = targets(tr)
R = tr.R.values.reshape(-1, 80)[:, 0]; C = tr.C.values.reshape(-1, 80)[:, 0]; U = tr.u_in.values.reshape(-1, 80); T = tr.time_step.values.reshape(-1, 80)
UO = tr.u_out.values.reshape(-1, 80)
va = fold_of == 0; tn = ~va
S = np.load(f'{ROOT}/runs/exp06/fold0_oof.npy')
E = np.mean([np.load(f'{ROOT}/runs/{r}/fold0_oof.npy') for r in ('exp04', 'exp06', 'exp07', 'exp08')], 0)
Yv, Mv = Y[va], M[va]
def bmae(P, Yx, Mx): e = np.abs(P - Yx); return (e * Mx).sum(1) / Mx.sum(1)
bS, bE = bmae(S, Yv, Mv), bmae(E, Yv, Mv)
say('fold-0 OOF: single', round(np.abs(S - Yv)[Mv].mean(), 5), 'ensemble4', round(np.abs(E - Yv)[Mv].mean(), 5), 'val breaths', va.sum())

# 1. by (R,C), incl. train-fold MAE of the single model (eval mode) to see where the gap is
code = open(f'{ROOT}/src/train_lstm_v2.py').read(); tree = ast.parse(code)
defs = {n.name: ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
g = dict(np=np, pd=pd, torch=torch, nn=nn); exec(defs['feats'], g)
ft = g['feats'](tr); FEATS = [c for c in ft.columns if c not in ('id', 'breath_id', 'pressure')]
X = ft[FEATS].values.astype('float32').reshape(-1, 80, len(FEATS)); del ft
a = json.load(open(f'{ROOT}/runs/exp06/fold0_args.json')); g['args'] = types.SimpleNamespace(**a); exec(defs['Net'], g)
net = g['Net'](len(FEATS)); net.load_state_dict(torch.load(f'{ROOT}/runs/exp06/fold0_best.pt', map_location='cpu')); net.eval()
mu, sd = np.load(f'{ROOT}/runs/exp06/fold0_scaler.npy'); Xt = (X[tn] - mu) / sd
with torch.no_grad(): Ptr = np.concatenate([net(torch.from_numpy(Xt[i:i + 1024])).numpy() for i in range(0, len(Xt), 1024)])
bTr = bmae(Ptr, Y[tn], M[tn])
rc = pd.DataFrame(dict(R=R[va], C=C[va], S=bS, E=bE, w=Mv.sum(1)))
rct = pd.DataFrame(dict(R=R[tn], C=C[tn], train=bTr, w=M[tn].sum(1)))
def wmean(df, col): return (df[col] * df.w).sum() / df.w.sum()
tab = rc.groupby(['R', 'C']).apply(lambda d: pd.Series(dict(n=len(d), single=wmean(d, 'S'), ens4=wmean(d, 'E'),
       share_of_err_single=(d.S * d.w).sum() / (rc.S * rc.w).sum())))
tab['avg_gain'] = tab.single - tab.ens4
tab['train_single'] = rct.groupby(['R', 'C']).apply(lambda d: wmean(d, 'train'))
tab['val_over_train'] = tab.single / tab.train_single
tab.round(4).to_csv(f'{OUT}/by_RC.csv'); say('\n[1] by (R,C), step-weighted MAE\n' + tab.round(4).to_string())

# 2. by position within inspiration
pos = np.cumsum(Mv, 1) - 1
rows = []
for k in range(int(pos[Mv].max()) + 1):
    sel = Mv & (pos == k)
    rows.append(dict(k=k, n=int(sel.sum()), single=np.abs(S - Yv)[sel].mean(), ens4=np.abs(E - Yv)[sel].mean()))
pk = pd.DataFrame(rows); pk.round(4).to_csv(f'{OUT}/by_insp_position.csv', index=False)
say('\n[2] by position within inspiration (k = 0 is the first u_out==0 step)\n' + pk.iloc[::3].round(4).to_string(index=False))

# 3. concentration
for name, b in (('single', bS), ('ens4', bE)):
    w = Mv.sum(1); e = b * w; o = np.argsort(-b); cs = np.cumsum(e[o]) / e.sum()
    say(f'\n[3] {name}: breath MAE median {np.median(b):.4f} p90 {np.percentile(b, 90):.4f} p99 {np.percentile(b, 99):.4f} max {b.max():.3f};'
        f' error share of worst 1% {cs[int(.01 * len(b)) - 1]:.3f}, 5% {cs[int(.05 * len(b)) - 1]:.3f}, 10% {cs[int(.10 * len(b)) - 1]:.3f}')

# 4. averaging gain by single-model difficulty decile
dec = pd.qcut(bS, 10, labels=False)
ag = pd.DataFrame(dict(dec=dec, S=bS, E=bE)).groupby('dec').agg(single=('S', 'mean'), ens4=('E', 'mean'))
ag['gain'] = ag.single - ag.ens4; ag['rel_gain'] = ag.gain / ag.single
ag.round(4).to_csv(f'{OUT}/avg_gain_by_decile.csv'); say('\n[4] averaging gain by single-model breath-difficulty decile (0 easiest)\n' + ag.round(4).to_string())

# 5. rarity: nearest training-fold breath (same R,C) in u_in space over the first 32 steps
L = 32; nn_d = np.zeros(va.sum()); vidx = np.where(va)[0]
for (r, c), _ in rc.groupby(['R', 'C']):
    mt = tn & (R == r) & (C == c); mv = (R[va] == r) & (C[va] == c)
    knn = NearestNeighbors(n_neighbors=1).fit(U[mt, :L]); d, _ = knn.kneighbors(U[vidx[mv], :L]); nn_d[mv] = d[:, 0]
rho = spearmanr(nn_d, bE)
say(f'\n[5] rarity: NN distance (u_in first {L} steps, same R,C) vs breath MAE (ens4): Spearman rho {rho.correlation:.3f} (p {rho.pvalue:.1e})')
rq = pd.DataFrame(dict(q=pd.qcut(nn_d, 10, labels=False), d=nn_d, E=bE)).groupby('q').agg(nn_dist=('d', 'median'), mae=('E', 'mean'), n=('E', 'size'))
rq.round(4).to_csv(f'{OUT}/mae_by_rarity_decile.csv'); say('MAE by NN-distance decile (9 = rarest)\n' + rq.round(4).to_string())
pct = pd.Series(nn_d).rank(pct=True).values
worst = np.argsort(-bE)[:20]; w1 = np.argsort(-bE)[:int(.01 * len(bE))]
say(f'NN-distance percentile: worst 20 median {np.median(pct[worst]):.2f}; worst 1% median {np.median(pct[w1]):.2f}; all 0.50 by construction')
say(f'worst 20 in rarest decile: {(pct[worst] > .9).sum()}/20; worst 1% in rarest decile: {(pct[w1] > .9).mean():.2f}')

# 6. worst-20 table and plots
vb = bid[va]
wt = pd.DataFrame(dict(breath_id=vb[worst], R=R[va][worst], C=C[va][worst], mae_ens4=bE[worst], mae_single=bS[worst], nn_dist=nn_d[worst], nn_pct=pct[worst],
                       u_in_max=U[va][worst].max(1), u_in_mean_insp=[U[va][i][Mv[i]].mean() for i in worst], p_max=Yv[worst].max(1)))
wt.round(4).to_csv(f'{OUT}/worst20.csv', index=False); say('\n[6] worst 20 breaths (ens4)\n' + wt.round(3).to_string(index=False))
fig, axs = plt.subplots(4, 5, figsize=(22, 15))
for ax, i in zip(axs.flat, worst):
    t = T[va][i]; ax.plot(t, Yv[i], 'k-', lw=2, label='true p'); ax.plot(t, E[i], 'C1-', label='ens4'); ax.plot(t, S[i], 'C0--', lw=1, label='exp06')
    ax2 = ax.twinx(); ax2.plot(t, U[va][i], 'C2:', label='u_in'); ax2.set_ylim(0, 100)
    out = UO[va][i] == 1
    if out.any(): ax.axvspan(t[out][0], t[-1], color='0.9', zorder=-1)
    ax.set_title(f'b{vb[i]} R{R[va][i]} C{C[va][i]} MAE {bE[i]:.2f} rar {pct[i]:.2f}', fontsize=9)
axs.flat[0].legend(fontsize=7, loc='upper right'); fig.suptitle('20 worst fold-0 breaths (4-model ensemble). black true, orange ens4, blue exp06, green dotted u_in (right axis 0-100); grey u_out==1')
fig.tight_layout(); fig.savefig(f'{OUT}/worst20.png', dpi=70)
say('\nwrote', OUT); log.close()
