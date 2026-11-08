#!/bin/bash
# FINAL-ENSEMBLE POD #1 (Sunday night 2026-11-08, 12 h dsmlp pod). APPROVED by the user 2026-11-08 ("Approve pod script"). Launch
# AFTER runs/sunqueue.txt shows "sunqueue done" and the DataHub server is stopped:
#   cd ~/ventilator-pressure-prediction && nohup bash src/podqueue_final1.sh > runs/podqueue_final1.txt 2>&1 &
# Config comes from the pre-registered lock (src/decide_lock.py). This script is written for the expected lock bs128/h128
# and refuses to start on any other lock (then a new plan is needed).
# Order fills missing folds first, so each seed becomes a complete 5-fold set: seed 43 (exp12b) f3-f4, seed 44 (exp12c) f1-f4, seed 45 (exp12d) f0-f1.
set -u
cd ~/ventilator-pressure-prediction
python src/decide_lock.py > runs/lock_stdout.txt 2>&1; LOCK=$(tail -1 runs/lock_stdout.txt)
echo "podqueue_final1 start $(date -u) :: $LOCK"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ "$LOCK" = "LOCK bs128/h128" ] || { echo "lock is not bs128/h128; refusing to start"; exit 1; }
A="--hidden 128 --layers 4 --epochs 300 --lr 2e-3 --bs 128"
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
[ -f runs/exp12b/fold1_test.npy ] || run exp12b 1 $A --seed 43   # only if today's run did not finish
[ -f runs/exp12b/fold2_test.npy ] || run exp12b 2 $A --seed 43
run exp12b 3 $A --seed 43
run exp12b 4 $A --seed 43
run exp12c 1 $A --seed 44
run exp12c 2 $A --seed 44
run exp12c 3 $A --seed 44
run exp12c 4 $A --seed 44
run exp12d 0 $A --seed 45
run exp12d 1 $A --seed 45
echo "podqueue_final1 done $(date -u)"; touch runs/PODQUEUE_FINAL1_DONE
