#!/bin/bash
# wait for exp02 fold 0, then train folds 1-4 one after another
cd ~/ventilator-pressure-prediction
while ! grep -q "FOLD 0 DONE" runs/exp02/stdout.txt; do sleep 30; done
for k in 1 2 3 4; do
  python src/train_lstm.py --name exp02 --folds $k --epochs 120 --bs 512 --lr 2e-3 > runs/exp02/stdout_fold$k.txt 2>&1
done
echo ALL DONE > runs/exp02/ALL_DONE
