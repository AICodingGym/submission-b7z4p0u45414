#!/bin/bash
# FINAL-ENSEMBLE POD #2 (Monday 2026-11-09, 12 h dsmlp pod). DRAFT - not approved yet. Launch only after the user approves,
# with the DataHub server stopped:
#   cd ~/ventilator-pressure-prediction && nohup bash src/podqueue_final2.sh > runs/podqueue_final2.txt 2>&1 &
# Same locked config bs128/h128 (refuses on any other lock). Only the seed and fold change.
# Order: finish seed 45 (exp12d f2-f4), then a full new seed 46 (exp12e f0-f4). 8 runs, about 11.1 h at 83 min/run.
# A run whose fold test file already exists is skipped, so a relaunch never repeats finished work.
set -u
cd ~/ventilator-pressure-prediction
python src/decide_lock.py > runs/lock_stdout.txt 2>&1; LOCK=$(tail -1 runs/lock_stdout.txt)
echo "podqueue_final2 start $(date -u) :: $LOCK"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
[ -f runs/exp12d/fold2_test.npy ] || run exp12d 2 $A --seed 45
[ -f runs/exp12d/fold3_test.npy ] || run exp12d 3 $A --seed 45
[ -f runs/exp12d/fold4_test.npy ] || run exp12d 4 $A --seed 45
[ -f runs/exp12e/fold0_test.npy ] || run exp12e 0 $A --seed 46
[ -f runs/exp12e/fold1_test.npy ] || run exp12e 1 $A --seed 46
[ -f runs/exp12e/fold2_test.npy ] || run exp12e 2 $A --seed 46
[ -f runs/exp12e/fold3_test.npy ] || run exp12e 3 $A --seed 46
[ -f runs/exp12e/fold4_test.npy ] || run exp12e 4 $A --seed 46
echo "podqueue_final2 done $(date -u)"; touch runs/PODQUEUE_FINAL2_DONE
