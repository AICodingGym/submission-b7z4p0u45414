#!/bin/bash
# Day queue 2026-11-06 on the DataHub GPU server (approved by the user). Order: T2 (150 epochs) then T1 (loss on all 80 steps).
# Waits for exp09 to finish first. Decision rules are in src/decide.py (pre-registered in EXPERIMENT_LOG.md).
set -u
cd ~/ventilator-pressure-prediction
COMMON="--folds 0 --bs 512 --lr 2e-3 --hidden 128 --layers 4"
echo "dayqueue start $(date -u)"
while [ ! -f runs/exp09/fold0_test.npy ] || pgrep -f "train_lstm_v2.py --name exp09" > /dev/null; do sleep 30; done
echo "$(date -u +%H:%M) exp09 finished :: $(tail -1 runs/exp09/stdout_fold0.txt)"
run () { local name=$1; shift; mkdir -p runs/$name; echo "$(date -u +%H:%M) START $name :: $*"
  python src/train_lstm_v2.py --name $name "$@" > runs/$name/stdout_fold0.txt 2>&1
  echo "$(date -u +%H:%M) END   $name :: $(tail -1 runs/$name/stdout_fold0.txt)"; }
run exp10a $COMMON --epochs 150 --seed 42
run exp10b $COMMON --epochs 150 --seed 43
EP=$(python src/decide.py T2 2>> runs/dayqueue.txt)
echo "$(date -u +%H:%M) T2 decision: T1 runs at $EP epochs"
run exp11a $COMMON --epochs $EP --loss_mask all --seed 42
run exp11b $COMMON --epochs $EP --loss_mask all --seed 43
echo "$(date -u +%H:%M) T1 decision: $(python src/decide.py T1 $EP 2>> runs/dayqueue.txt)"
echo "dayqueue done $(date -u)"; touch runs/DAYQUEUE_DONE
