#!/bin/bash
# SATURDAY DAY QUEUE (2026-11-07), on the DataHub server (container started 19:36 UTC; planned stop 01:36 UTC, 6 h, UNCONFIRMED).
#   cd ~/ventilator-pressure-prediction && nohup bash src/satqueue.sh > runs/satqueue.txt 2>&1 &
# No CPU jobs while this runs (bs 64 measured +51% slower with an 8-thread CPU job alongside).
set -u
cd ~/ventilator-pressure-prediction
A="--hidden 128 --layers 4 --epochs 300 --lr 2e-3"
echo "satqueue start $(date -u)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run () { local name=$1 fold=$2; shift 2; mkdir -p runs/$name; echo "$(date -u +%H:%M) START $name fold $fold :: $*"
  python src/train_lstm_v2.py --name $name --folds $fold "$@" > runs/$name/stdout_fold$fold.txt 2>&1
  echo "$(date -u +%H:%M) END   $name fold $fold :: $(tail -1 runs/$name/stdout_fold$fold.txt)"; }
run exp16a 0 $A --bs 64 --seed 42     # D1: bs 128 -> 64 (vs new reference N1 = exp12a/b/c), LR unchanged
run exp16b 0 $A --bs 64 --seed 43
run exp12a 1 $A --bs 128 --seed 42    # reference config (N1, seed 42) on fold 1: ensemble coverage + replication of N1 on a 2nd fold
echo "satqueue done $(date -u)"; touch runs/SATQUEUE_DONE
