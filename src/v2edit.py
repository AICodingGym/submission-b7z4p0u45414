import sys
f=sys.argv[1]; s=open(f).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a[:50]
    s=s.replace(a,b)
rep("(v1 fit them on all train rows incl. the validation fold). Folds come from common.py (same logic as v1).\n",
    "(v1 fit them on all train rows incl. the validation fold). Folds come from common.py (same logic as v1).\n"
    "v2 bookkeeping (no effect on training): --wd flag (default 1e-5 = v1 value); per-fold args file with script md5;\n"
    "test predictions saved to fold{k}_test_best.npy every time validation improves (survives a container stop);\n"
    "fold{k}_test.npy (= 'fold completed' marker used by ensemble.py) is still written only after the last epoch.\n")
rep("p.add_argument('--dropout', type=float, default=0.0)\n",
    "p.add_argument('--dropout', type=float, default=0.0)\np.add_argument('--wd', type=float, default=1e-5)      # AdamW weight decay (v1 hard-coded 1e-5)\n")
rep("import sys\n", "import sys, hashlib\n")
rep("json.dump(vars(args), open(os.path.join(OUT, 'args.json'), 'w'), indent=1)\n",
    "SCRIPT_MD5 = hashlib.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()\n")
rep("""def predict(model, X):""", """def submission_order(pt):
    # predictions come out in (breath_id, time_step) order; put them back in test.csv id order
    return pd.DataFrame({'id': te.id.values, 'pressure': pt.reshape(-1)}).set_index('id').loc[te_order].pressure.values

def predict(model, X):""")
rep("""    va = fold_of == k; trn = ~va
""", """    va = fold_of == k; trn = ~va
    json.dump(dict(vars(args), fold=k, script=os.path.basename(__file__), script_md5=SCRIPT_MD5),
              open(f'{OUT}/fold{k}_args.json', 'w'), indent=1)
""")
rep("    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)\n",
    "    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)\n")
rep("""            best = vmae; torch.save(model.state_dict(), f'{OUT}/fold{k}_best.pt'); np.save(f'{OUT}/fold{k}_oof.npy', pv)
""", """            best = vmae; torch.save(model.state_dict(), f'{OUT}/fold{k}_best.pt'); np.save(f'{OUT}/fold{k}_oof.npy', pv)
            np.save(f'{OUT}/fold{k}_test_best.npy', submission_order(predict(model, Xte)))   # best-so-far test preds
""")
rep("""    s = pd.DataFrame({'id': te.id.values, 'pressure': pt.reshape(-1)}).set_index('id').loc[te_order].reset_index()
    np.save(f'{OUT}/fold{k}_test.npy', s.pressure.values)
""", """    s = pd.DataFrame({'id': te_order, 'pressure': submission_order(pt)})
    np.save(f'{OUT}/fold{k}_test.npy', s.pressure.values)
""")
open(f,"w").write(s)
# --- part 2: smoke-test support (no effect on normal runs) ---
s=open(f).read()
rep("p.add_argument('--wd', type=float, default=1e-5)      # AdamW weight decay (v1 hard-coded 1e-5)\n",
    "p.add_argument('--wd', type=float, default=1e-5)      # AdamW weight decay (v1 hard-coded 1e-5)\n"
    "p.add_argument('--debug_breaths', type=int, default=0)  # >0: smoke test on this many train/val breaths only\n")
rep("dev = 'cuda'\n", "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n")
rep("""    va = fold_of == k; trn = ~va
""", """    va = fold_of == k; trn = ~va
    if args.debug_breaths:
        trn = trn & (np.cumsum(trn) <= args.debug_breaths); va = va & (np.cumsum(va) <= args.debug_breaths)
""")
open(f,"w").write(s)
