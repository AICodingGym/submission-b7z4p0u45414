# PA3 Experiment Log: Ventilator Pressure Prediction

AI assistant: Claude Code (Opus 5.5), driving DataHub through the browser. Rules given to the AI: no existing solutions, no original Kaggle data.
Validation: 5-fold split by breath_id, stratified by (R,C). Metric = MAE on u_out==0 steps only (matches the leaderboard).

## Overnight plan (12 h container, v2 script, all fold-0 unless noted; ~85 min per run = 300 x ~16 s + prep, GUESS for a new node)
Choose A if exp05's final fold-0 OOF < exp04's 0.21621, otherwise B. The decision rule for folds 1-4 is pre-registered in overnight.sh.
| plan | run | single change (vs) | hypothesis | expected fold-0 OOF (GUESS) | proven wrong if | GPU |
|---|---|---|---|---|---|---|
| A | exp06 | v2 scalers (vs exp05) | the scaler leak was negligible | exp05 +/- 0.005 | differs by > 0.005 (unless noise from exp08 is that big) | 85 min |
| A | exp07 | dropout 0.1 -> 0.2 (vs exp06) | more dropout narrows the gap further | exp06 - 0.005 | >= exp06 - noise | 85 min |
| A | exp08 | seed 42 -> 43 (vs exp06) | measures seed noise | exp06 +/- 0.005 | (measurement, not a test) | 85 min |
| A | best of exp06/07 folds 1-4 | fold (vs its fold 0) | gives a 5-fold model for the test ensemble | each fold within +/- 0.01 of fold 0 | any fold > 0.02 off (would suggest a fold problem) | 4 x 85 min |
| B | exp06 | v2 scalers (vs exp04) | the scaler leak was negligible | 0.2162 +/- 0.005 | differs by > 0.005 | 85 min |
| B | exp07 | weight decay 1e-5 -> 1e-2 (vs exp06) | decoupled weight decay narrows the gap where dropout did not | 0.205-0.215 | >= exp06 - noise | 85 min |
| B | exp08 | seed 42 -> 43 (vs exp06) | measures seed noise | exp06 +/- 0.005 | (measurement) | 85 min |
| B | best of exp06/07 folds 1-4 | fold | 5-fold model | +/- 0.01 of fold 0 | any fold > 0.02 off | 4 x 85 min |
Total per plan: 7 x 85 min = ~9.9 h of an 11 h budget.

## Data facts (EDA, 2026-11-05)
- train 5,432,400 rows = 67,905 breaths x 80 steps; test 603,600 rows = 7,545 breaths. No breath overlap. No nulls.
- 9 (R,C) combos; test distribution matches train. ~38% of steps are inspiratory (u_out==0), 25-32 per breath.
- dt is about 0.033 s; one breath is about 2.94 s.

## Experiments
| id | change (one at a time) | holdout insp MAE | LB | notes |
|---|---|---|---|---|
| B0 | global median of insp pressure | 7.538 | - | 20% breath holdout |
| B1 | median by (R,C,step) | 6.083 | - | ignores u_in entirely |
| exp01 | BiLSTM 4x256, 31 feats, masked L1, bs256 lr1e-3 | (killed ep2: 1.452) | - | 33 s/epoch, too slow for 5 folds in a 6 h container |
| exp02 | exp01 + bs512, lr2e-3, 120 ep, fold 0 | 0.2419 | 0.23918 | 29 s/ep. Val still falling when the LR schedule ended (undertrained). Train 0.144 vs val 0.242 (overfitting gap). Folds 1-4 cancelled. LB matches val, so CV is trustworthy. |
| exp03 | hidden 192, 3 layers, 300 ep | (killed ep1) | - | 70 s/ep. Slower than the bigger model! |
| exp04 | hidden 128, 4 layers, 300 ep, fold 0 | 0.2162 | - | 15 s/ep. Better than exp02 (0.2419). Gap grew: train 0.096 vs val 0.216, so overfitting is now the main problem. Grid snap on OOF: 0.2150 (-0.001). |
| exp05 | exp04 + LSTM inter-layer dropout 0.1 | running (mid-run check below) | | Pre-registered LATE (at epoch 128 of 300, best so far 0.2944, before the final result). Hypothesis: dropout narrows the real 2.2x train/val gap. Expected OOF: 0.20 (GUESS). GPU: ~80 min. Proven wrong if OOF >= 0.216 (no better than exp04) or if eval-mode train MAE / val MAE stays >= 2x. |

### exp04 OOF error analysis
- Per-breath MAE: median 0.127, p90 0.41, p99 1.53. The worst 5% of breaths account for 30% of total error (heavy tail).
- By lung: R=50 is worst (0.23-0.31); R=5 with C=20/50 is best (0.13).
- Error grows over the breath: 0.16 at step 0 to 0.24 mid-breath to 0.29 at the end of inspiration.

## AI decisions / mistakes caught
- MISTAKE: the AI picked 120 epochs to fit 5 folds in the 6 h container, without checking convergence. The exp02 curve was still improving at the end. Lesson: read the learning curve before scaling to more folds.
- MISTAKE: the AI picked hidden=192 to get a "smaller, faster" model. It was 2.1x slower per step than 256 (cuDNN slow path for a size that isn't a power of two). Caught because the epoch time went UP. A microbenchmark (ms per fwd+bwd step, bs512): 256x4 254, 192x3 532, 256x3 184, 128x4 121, 128x3 90. Lesson: benchmark instead of assuming.
- DATA FINDING: pressure takes exactly 950 values on a 0.0703 grid (found via np.unique on train). Possible post-processing: snap to the grid. Test on OOF before using.
- Features were chosen from physics: pressure is about R*flow + volume/C, so the AI added vol = cumsum(u_in*dt), vol/C, u_in*R, lags/leads. (AI design choice; check by ablation.)
- The AI assumed a bigger batch would fix the slow epochs. Measuring GPU utilization (97%) showed the model itself is the bottleneck, so the bigger batch helped only ~10%.
- Loss is masked to u_out==0 because that is what is scored (from the metric definition, not a competition trick).
