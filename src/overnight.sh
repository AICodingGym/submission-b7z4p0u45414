#!/bin/bash
# Overnight queue for PA3 (v2 script). Usage:  bash src/overnight.sh A   or   bash src/overnight.sh B
#   A = use if exp05 (dropout 0.1) beat exp04 on fold-0 OOF;  B = use if it did not.
# 7 runs x ~85 min (300 epochs x ~16 s + data prep + test saves) = ~9.9 h, inside an 11 h budget. Timing is measured
# on the current container's 1080 Ti (15.1-15.8 s/epoch); a different node may differ (GUESS that it won't by much).
# Pre-registered decision rule for folds 1-4 (applied automatically after run 3):
#   noise = |seed-repeat OOF - baseline OOF|; use the challenger only if challenger < baseline - noise, else the baseline.
set -u
cd ~/ventilator-pressure-prediction
PLAN=${1:?usage: overnight.sh A|B}
COMMON="--epochs 300 --bs 512 --lr 2e-3 --hidden 128 --layers 4"
if [ "$PLAN" = A ]; then
  BASE="exp06"; BASE_ARGS="$COMMON --dropout 0.1"                 # = exp05 config on v2 (scaler fix only)
  CHAL="exp07"; CHAL_ARGS="$COMMON --dropout 0.2"                 # one change vs exp06: dropout 0.1 -> 0.2
  SEED="exp08"; SEED_ARGS="$COMMON --dropout 0.1 --seed 43"       # one change vs exp06: seed 42 -> 43
elif [ "$PLAN" = B ]; then
  BASE="exp06"; BASE_ARGS="$COMMON"                               # = exp04 config on v2 (scaler fix only)
  CHAL="exp07"; CHAL_ARGS="$COMMON --wd 1e-2"                     # one change vs exp06: weight decay 1e-5 -> 1e-2
  SEED="exp08"; SEED_ARGS="$COMMON --seed 43"                     # one change vs exp06: seed 42 -> 43
else
  echo "plan must be A or B"; exit 1
fi
echo "plan $PLAN start $(date -u)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run () {  # run <name> <fold> <args...>
  local name=$1 fold=$2; shift 2
  mkdir -p runs/$name
  echo "$(date -u +%H:%M) START $name fold $fold :: $*"
  python src/train_lstm_v2.py --name $name --folds $fold "$@" > runs/$name/stdout_fold$fold.txt 2>&1
  echo "$(date -u +%H:%M) END   $name fold $fold :: $(tail -1 runs/$name/stdout_fold$fold.txt)"
}

run $BASE 0 $BASE_ARGS
run $CHAL 0 $CHAL_ARGS
run $SEED 0 $SEED_ARGS

PICK=$(python - "$BASE" "$CHAL" "$SEED" <<'EOF'
import sys, pandas as pd
b, c, s = sys.argv[1:]
best = lambda r: pd.read_csv(f'runs/{r}/fold0_log.csv').val_mae.min()
try:
    B, C, S = best(b), best(c), best(s)
except Exception as e:
    print(b); sys.stderr.write(f'pick: missing log ({e}); defaulting to baseline\n'); sys.exit()
noise = abs(S - B)
choice = c if C < B - noise else b
sys.stderr.write(f'pick: base {B:.5f} chal {C:.5f} seed {S:.5f} noise {noise:.5f} -> {choice}\n')
print(choice)
EOF
)
if [ "$PICK" = "$CHAL" ]; then PICK_ARGS=$CHAL_ARGS; else PICK_ARGS=$BASE_ARGS; fi
echo "$(date -u +%H:%M) folds 1-4 use $PICK :: $PICK_ARGS"
for k in 1 2 3 4; do run $PICK $k $PICK_ARGS; done
echo "plan $PLAN done $(date -u)"; touch runs/OVERNIGHT_${PLAN}_DONE
