#!/bin/bash
# EVENING QUEUE, DRAFT: NOT approved. The user approves it after T2's verdict. Run on a 12 h dsmlp pod: bash src/evening.sh
# Reads runs/decision_T2.json (epochs) and runs/decision_T1.json (T1 verdict). Every challenger runs seeds 42 and 43.
set -u
cd ~/ventilator-pressure-prediction
EP=$(python -c "import json; print(150 if json.load(open('runs/decision_T2.json')).get('passed') else 300)")
T1=$(python -c "import json; print('ADOPT' if json.load(open('runs/decision_T1.json')).get('adopted') else 'REJECT')" 2>/dev/null || echo NONE)
ARCH="--folds 0 --hidden 128 --layers 4 --epochs $EP"
echo "evening start $(date -u) :: epochs $EP, T1 $T1"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run () { local name=$1; shift; mkdir -p runs/$name; echo "$(date -u +%H:%M) START $name :: $*"
  python src/train_lstm_v2.py --name $name "$@" > runs/$name/stdout_fold0.txt 2>&1
  echo "$(date -u +%H:%M) END   $name :: $(tail -1 runs/$name/stdout_fold0.txt)"; }
# N1: batch 512 -> 128, LR unchanged (2e-3)
run exp12a $ARCH --bs 128 --lr 2e-3 --seed 42
run exp12b $ARCH --bs 128 --lr 2e-3 --seed 43
# N2: weight decay 1e-5 -> 1e-2 (AdamW). At 300 epochs, exp07 already is seed 42.
if [ "$EP" = 150 ]; then run exp13a $ARCH --bs 512 --lr 2e-3 --wd 1e-2 --seed 42; fi
run exp13b $ARCH --bs 512 --lr 2e-3 --wd 1e-2 --seed 43
# N3: third control seed at 150 epochs (at 300, exp09 already is seed 44)
if [ "$EP" = 150 ]; then run exp14 $ARCH --bs 512 --lr 2e-3 --seed 44; fi
# N5: if T1 was adopted, a third T1 seed
if [ "$T1" = ADOPT ]; then run exp11c $ARCH --bs 512 --lr 2e-3 --loss_mask all --seed 44; fi
echo "evening done $(date -u)"; touch runs/EVENING_DONE
