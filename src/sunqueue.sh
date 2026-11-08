#!/bin/bash
# SUNDAY DAY QUEUE (2026-11-08), DataHub server (container started 18:53 UTC; planned stop 00:53 UTC, UNCONFIRMED).
# Per the pre-registered Sunday plan: D1 (bs64) is the only arm with diff < 0 (-0.0006) -> its seed 44; H1 (+0.0101) is not close.
# Remaining time -> reference-config folds (bs128/h128, seed 43 = exp12b).
#   cd ~/ventilator-pressure-prediction && nohup bash src/sunqueue.sh > runs/sunqueue.txt 2>&1 &
set -u
cd ~/ventilator-pressure-prediction
A="--hidden 128 --layers 4 --epochs 300 --lr 2e-3"
echo "sunqueue start $(date -u)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
run exp16c 0 $A --bs 64 --seed 44     # D1 third seed
run exp12b 1 $A --bs 128 --seed 43    # reference folds
run exp12b 2 $A --bs 128 --seed 43
echo "sunqueue done $(date -u)"; touch runs/SUNQUEUE_DONE
