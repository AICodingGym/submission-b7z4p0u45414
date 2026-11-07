#!/bin/bash
# SATURDAY NIGHT POD QUEUE (12 h dsmlp pod), DRAFT 2026-11-07. Launch only after the user approves, AFTER the DataHub server is stopped:
#   cd ~/ventilator-pressure-prediction && nohup bash src/podqueue_sat.sh > runs/podqueue_sat.txt 2>&1 &
# The reference batch size is chosen by the pre-registered D1 rule (bs 64 only if D1 is ADOPTED; otherwise bs 128).
set -u
cd ~/ventilator-pressure-prediction
D1=$(python src/decide_vs_ref.py D1 exp16a exp16b)
[ "$D1" = ADOPT ] || [ "$D1" = REJECT ] || { echo "D1 incomplete ($D1); refusing to start"; exit 1; }
if [ "$D1" = ADOPT ]; then BS=64; REF=exp16a; else BS=128; REF=exp12a; fi
A="--hidden 128 --layers 4 --epochs 300 --lr 2e-3 --bs $BS"
H="--hidden 256 --layers 4 --epochs 300 --lr 2e-3 --bs $BS"
echo "podqueue start $(date -u) :: D1 $D1 -> bs $BS, fold runs extend $REF (seed 42)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run () { local name=$1 fold=$2; shift 2; mkdir -p runs/$name; echo "$(date -u +%H:%M) START $name fold $fold :: $*"
  python src/train_lstm_v2.py --name $name --folds $fold "$@" > runs/$name/stdout_fold$fold.txt 2>&1
  echo "$(date -u +%H:%M) END   $name fold $fold :: $(tail -1 runs/$name/stdout_fold$fold.txt)"; }
run exp18a 0 $H --seed 42      # H1: hidden 128 -> 256 at the reference batch size
run exp18b 0 $H --seed 43
[ -f runs/exp12a/fold1_test.npy ] || [ $BS = 64 ] || run exp12a 1 $A --seed 42   # only if today's fold-1 run did not finish
[ $BS = 64 ] && run $REF 1 $A --seed 42
run $REF 2 $A --seed 42        # reference config, more folds: final-ensemble coverage
run $REF 3 $A --seed 42
run $REF 4 $A --seed 42
echo "podqueue done $(date -u)"; touch runs/PODQUEUE_SAT_DONE
