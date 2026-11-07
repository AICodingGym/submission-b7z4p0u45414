#!/bin/bash
# SATURDAY NIGHT POD QUEUE (12 h dsmlp pod). APPROVED by the user 2026-11-07. Launch AFTER runs/satqueue.txt shows "satqueue done"
# and AFTER the DataHub server is stopped (one GPU per user):
#   cd ~/ventilator-pressure-prediction && nohup bash src/podqueue_sat.sh > runs/podqueue_sat.txt 2>&1 &
# The reference batch size is chosen by the pre-registered D1 rule (bs 64 only if D1 is ADOPTED; otherwise bs 128).
# Every run logs its measured seconds/epoch after its first 10 timed epochs (epochs 1-10; epoch 0 includes warm-up).
set -u
cd ~/ventilator-pressure-prediction
D1=$(python src/decide_vs_ref.py D1 exp16a exp16b)
[ "$D1" = ADOPT ] || [ "$D1" = REJECT ] || { echo "D1 incomplete ($D1); refusing to start"; exit 1; }
if [ "$D1" = ADOPT ]; then BS=64; REF=exp16a; else BS=128; REF=exp12a; fi
A="--hidden 128 --layers 4 --epochs 300 --lr 2e-3 --bs $BS"
H="--hidden 256 --layers 4 --epochs 300 --lr 2e-3 --bs $BS"
echo "podqueue start $(date -u) :: D1 $D1 -> bs $BS, fold runs extend $REF (seed 42)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
speed () { local f=$1; grep "^fold [0-9] ep " "$f" | sed -n 2,11p | awk '{s=$NF; sub(/s$/,"",s); print s}' | sort -n \
  | awk '{a[NR]=$1} END{ if (NR<10) {print "NA"; exit}; m=(a[5]+a[6])/2; printf "median %.1f s/epoch over epochs 1-10 -> projected %.0f min for 300 epochs\n", m, m*300/60 }'; }
run () { local name=$1 fold=$2; shift 2; mkdir -p runs/$name; local out=runs/$name/stdout_fold$fold.txt
  echo "$(date -u +%H:%M) START $name fold $fold :: $*"
  python src/train_lstm_v2.py --name $name --folds $fold "$@" > $out 2>&1 &
  local pid=$! logged=0
  while kill -0 $pid 2>/dev/null; do
    if [ $logged = 0 ] && [ "$(grep -c "^fold [0-9] ep " $out 2>/dev/null)" -ge 11 ]; then
      echo "$(date -u +%H:%M) SPEED $name fold $fold :: $(speed $out)"; logged=1; fi
    sleep 30; done
  wait $pid
  [ $logged = 1 ] || echo "$(date -u +%H:%M) SPEED $name fold $fold :: $(speed $out)"
  echo "$(date -u +%H:%M) END   $name fold $fold :: $(tail -1 $out)"; }
run exp18a 0 $H --seed 42      # H1: hidden 128 -> 256 at the reference batch size
run exp18b 0 $H --seed 43
[ -f runs/exp12a/fold1_test.npy ] || [ $BS = 64 ] || run exp12a 1 $A --seed 42   # only if today's fold-1 run did not finish
[ $BS = 64 ] && run $REF 1 $A --seed 42
run $REF 2 $A --seed 42        # reference config, more folds: final-ensemble coverage
run $REF 3 $A --seed 42
run $REF 4 $A --seed 42
echo "podqueue done $(date -u)"; touch runs/PODQUEUE_SAT_DONE
