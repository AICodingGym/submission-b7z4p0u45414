#!/bin/bash
# EVENING QUEUE, FINAL (2026-11-06). 300 epochs (T2 failed: 150 ep was +0.0346 worse). Run on a 12 h dsmlp pod AFTER the day queue:
#   cd ~/ventilator-pressure-prediction && nohup bash src/evening.sh > runs/evening.txt 2>&1 &
# Order = priority: the 2-seed arms first, then the seed-44 additions, then the T1 third seed (only if T1 was adopted).
set -u
cd ~/ventilator-pressure-prediction
[ -f runs/DAYQUEUE_DONE ] || { echo "day queue not finished; refusing to start"; exit 1; }
T1=$(python -c "import json; print('ADOPT' if json.load(open('runs/decision_T1.json')).get('adopted') else 'REJECT')" 2>/dev/null || echo NONE)
A="--folds 0 --hidden 128 --layers 4 --epochs 300 --lr 2e-3"
echo "evening start $(date -u) :: T1 $T1"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run () { local script=$1 name=$2; shift 2; mkdir -p runs/$name; echo "$(date -u +%H:%M) START $name ($script) :: $*"
  python src/$script --name $name "$@" > runs/$name/stdout_fold0.txt 2>&1
  echo "$(date -u +%H:%M) END   $name :: $(tail -1 runs/$name/stdout_fold0.txt)"; }
run train_lstm_v2.py exp12a $A --bs 128 --seed 42                 # N1: batch 512 -> 128, LR unchanged
run train_lstm_v2.py exp12b $A --bs 128 --seed 43
run train_lstm_v2.py exp13b $A --bs 512 --wd 1e-2 --seed 43      # N2: wd 1e-5 -> 1e-2 (exp07 = seed 42)
run train_lstm_v3.py exp15a $A --bs 512 --uin_noise 0.01 --seed 42   # N4: 1% multiplicative u_in noise (v3)
run train_lstm_v3.py exp15b $A --bs 512 --uin_noise 0.01 --seed 43
run train_lstm_v2.py exp12c $A --bs 128 --seed 44                 # N1 seed 44
run train_lstm_v2.py exp13c $A --bs 512 --wd 1e-2 --seed 44      # N2 seed 44
if [ "$T1" = ADOPT ]; then run train_lstm_v2.py exp11c $A --bs 512 --loss_mask all --seed 44; fi   # N5
echo "evening done $(date -u)"; touch runs/EVENING_DONE
