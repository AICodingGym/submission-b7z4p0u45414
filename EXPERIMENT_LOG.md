# PA3 Experiment Log: Ventilator Pressure Prediction

AI assistant: Claude Code (Opus 5.5), driving DataHub through the browser. Rules given to the AI: no existing solutions, no original Kaggle data.
Validation: 5-fold split by breath_id, stratified by (R,C). Metric = MAE on u_out==0 steps only (matches the leaderboard).

> Log integrity note (2026-11-06 ~00:10 UTC): this file was REBUILT by the AI. Its update helper had read the file through the browser HTTP cache, so several saves were built on a stale copy and silently overwrote earlier sections (Ground rules, Audit 1, Leaderboard submissions, Script versions, and some mistakes). Git commits made by the CLI did not have them either. All text is restored verbatim from the AI conversation, which is the primary record. The pre-rebuild file is kept as EXPERIMENT_LOG.partial_backup.md. From now on every save is followed by an uncached re-read check.

## Ground rules (set by me, 2026-11-05)
- Before any experiment the AI states: hypothesis, the ONE thing that changes, expected fold-0 OOF MAE (a guess), GPU time, and what result would prove it wrong.
- Every result is logged here. Guesses are labeled as guesses. Ideas that might be specific to this competition are flagged [COMPETITION-SPECIFIC?].
- The AI never submits, deletes, or kills a job without asking me.

## Leaderboard submissions (all via aicodinggym CLI by the AI, each after my explicit OK in chat)
| when (UTC) | what | OOF fold 0 | LB | evidence |
|---|---|---|---|---|
| 2026-11-05 21:21:46 | exp02 fold 0 (runs/exp02/fold0_submission.csv) | 0.24189 | 0.23918 | CLI printed "Score: 0.23918"; git commit dd78419 "MLE submission" at 21:21:46. Approved by me ("Yes, do all three") |
| 2026-11-05 23:47:31 | exp04 fold 0, submission.csv md5 671554ad... | 0.21621 | 0.21326 | CLI printed "Score: 0.21326"; git ad16696. Approved by me ("OK to submit the exp04 fold-0 submission.csv") |
| 2026-11-06 00:03:06 | MEAN+snap of exp02+exp04 fold 0 (submissions/sub03_mean_snap_exp02_exp04.csv, md5 b96d9e56...) | 0.21126 | 0.20797 | CLI "Score: 0.20797"; git 2138f82. Approved by me. Checks: 603,600 rows, header ok, ids in order, 0 NaN, 100% of values on the grid; 1 value clipped at the grid max (64.859 -> 64.821) |
- LB was 0.003 better than OOF on both single-model submissions, so fold-0 OOF is a slightly pessimistic but consistent proxy (2 data points).
| 2026-11-06 00:31:02 | MEAN no-snap of exp02+exp04 fold 0 (submissions/sub04_mean_nosnap_exp02_exp04.csv, md5 63efab2e...) | 0.21251 | 0.20926 | CLI "Score: 0.20926"; git 1650fb7. Approved by me ("Optional, if cheap: submit..."). Identical to sub03 except no snapping |
- SNAP ISOLATED (sub04 vs sub03): LB 0.20926 -> 0.20797 = -0.00129; OOF 0.21251 -> 0.21126 = -0.00125. Snapping helps by about the same amount on the LB as on OOF. [COMPETITION-SPECIFIC?] (see Data facts)
- (Superseded) Snap on LB cannot be isolated: sub03 differs from the exp04 submission in two ways (adds exp02 AND snaps). LB gain vs exp04 = 0.00529; OOF gain = 0.00495 (mean -0.00370, snap -0.00125). Isolating snap on the LB would need a MEAN-without-snap submission.

## Data facts (EDA, 2026-11-05)
- train 5,432,400 rows = 67,905 breaths x 80 steps; test 603,600 rows = 7,545 breaths. No breath overlap. No nulls.
- 9 (R,C) combos; test distribution matches train. ~38% of steps are inspiratory (u_out==0), 25-32 per breath.
- dt is about 0.033 s; one breath is about 2.94 s.
- Pressure takes exactly 950 values on a 0.0703 grid (np.unique on train). [COMPETITION-SPECIFIC?]: found by our own analysis, but it may be a known trick for this competition, so disclose how it was found.

## Experiments
| id | change (one at a time) | holdout insp MAE | LB | notes |
|---|---|---|---|---|
| B0 | global median of insp pressure | 7.538 | - | 20% breath holdout |
| B1 | median by (R,C,step) | 6.083 | - | ignores u_in entirely |
| exp01 | BiLSTM 4x256, 31 feats, masked L1, bs256 lr1e-3 | (killed ep2: 1.452) | - | 33 s/epoch, too slow for 5 folds in a 6 h container |
| exp02 | exp01 + bs512, lr2e-3, 120 ep, fold 0 | 0.2419 | 0.23918 | 29 s/ep. Val still falling when the LR schedule ended (undertrained). Train 0.144 vs val 0.242 (overfitting gap). Folds 1-4 cancelled. LB matches val, so CV is trustworthy. |
| exp03 | hidden 192, 3 layers, 300 ep | (killed ep1) | - | 70 s/ep. Slower than the bigger model! |
| exp04 | hidden 128, 4 layers, 300 ep, fold 0 | 0.2162 | 0.21326 | 15 s/ep. Better than exp02 (0.2419). Gap grew: train 0.096 vs val 0.216, so overfitting is now the main problem. Grid snap on OOF: 0.2150 (-0.001). |
| exp05 | exp04 + LSTM inter-layer dropout 0.1 | 0.21830 (best ep285) | - | Pre-registered LATE (at epoch 128 of 300, best so far 0.2944, before the final result). Hypothesis: dropout narrows the real 2.2x train/val gap. Expected OOF: 0.20 (GUESS). GPU: ~80 min. Proven wrong if OOF >= 0.216 (no better than exp04) or if eval-mode train MAE / val MAE stays >= 2x. |

### exp04 OOF error analysis
- Per-breath MAE: median 0.127, p90 0.41, p99 1.53. The worst 5% of breaths account for 30% of total error (heavy tail).
- By lung: R=50 is worst (0.23-0.31); R=5 with C=20/50 is best (0.13).
- Error grows over the breath: 0.16 at step 0 to 0.24 mid-breath to 0.29 at the end of inspiration.

### exp05 mid-run check (2026-11-06 ~00:05 UTC, epoch 206 of 300)
Best-so-far val (cummin) at the same epochs, exp04 / exp05: ep125 0.2873 / 0.2944; ep150 0.2559 / 0.2923; ep175 0.2407 / 0.2559; ep200 0.2278 / 0.2407; ep206 0.2278 / 0.2376. At ep222 exp05 was 0.2313 (exp04 at ep225: 0.2224).
- CORRECTED 2026-11-06 00:35 (the original sentence was wrong; see mistakes). Best-so-far gaps (exp05 - exp04) from the raw logs: ep25 -0.0825, ep50 -0.0151, ep75 +0.0066, ep100 -0.0017, ep125 +0.0071, ep150 +0.0364, ep175 +0.0152, ep200 +0.0128, ep206 +0.0098, ep222 +0.0077, ep250 +0.0049, ep275 +0.0023, ep299 +0.0021. exp05 was ahead at ep25/50/100 and behind at ep75. The gap peaked at ep150 and shrank steadily to the end; it was NOT a steady 0.009-0.010.
- The repeated-looking values are real coincidences at 4 decimals: exp04 ep145 val 0.25589842 vs exp05 ep159 0.25586593; exp04 ep175 0.24069080 vs exp05 ep200 0.24065445. 0 identical values at full precision between the two logs; train columns never equal; different file sizes.
- exp04 improved 0.0116 (5.1%) from ep206 to ep299 as OneCycle LR annealed. If exp05 gains the same 5.1% (GUESS), it ends near 0.225: worse than exp04 (0.2162) and far from the pre-registered 0.20.
- Verdict so far: ON COURSE TO FAIL the pre-registration. Reaching 0.20 would need a 16% drop in the last 94 epochs vs exp04's 5%.
- A fair final verdict needs: (1) all 300 epochs (the LR anneals to ~0 at the end, where exp04 got most of its late gain); (2) an eval-mode train MAE on the best checkpoint for the gap criterion (the logged train number includes dropout); (3) a seed-noise estimate. We have never measured seed-to-seed variance, so a difference under ~0.005 between any two single runs is not interpretable yet (GUESS at the noise size). The overnight queue includes a seed repeat for this reason.

### exp05 config / timing audit
- args.json exp04 vs exp05: identical except dropout (absent in exp04 = 0.0; 0.1 in exp05). Same epochs 300, bs 512, lr 2e-3, hidden 128, layers 4, loss_mask insp, seed 42, nfolds 5. So exp05 IS a one-change config experiment.
- Non-config differences, logged for honesty: (a) the script file changed between runs (the --dropout flag was added; behaviour at dropout=0 is identical, but exp05 ran on a different file revision with no md5 recorded); (b) the AI's CPU audit ran on the same container during exp05 (time cost 15.07 -> 15.71 s/epoch; it should not affect results); (c) cuDNN LSTM is not bit-deterministic, so even an identical config would not reproduce exactly.
- 15.8 s vs 29-33 s is NOT an exp05 change: 29-33 s were exp01/exp02 with hidden 256. exp04 (hidden 128) already ran at 15.1 s.
- Provenance issue found: runs/exp02/args.json says folds "1" because the killed fold-1 run overwrote the fold-0 args. Fold-0 settings are otherwise identical. v2 writes per-fold args files with the script md5 to prevent this.

### exp05 FINAL (2026-11-06 ~00:25 UTC)
- Raw last lines: exp04 "fold 0 ep 299 train 0.0962 val_insp_mae 0.2162 best 0.2162", "FOLD 0 DONE best val_insp_mae 0.21621" (best at ep273). exp05 "fold 0 ep 299 train 0.1580 val_insp_mae 0.2187 best 0.2183", "FOLD 0 DONE best val_insp_mae 0.21830" (best at ep285).
- Eval-mode, best checkpoint, scored steps, same 5,000 train-fold breaths (RandomState(1)) and all 13,584 fold-0 breaths: exp04 TRAIN 0.09912 / VAL 0.21621 (ratio 2.18); exp05 TRAIN 0.12429 / VAL 0.21830 (ratio 1.76).
- Verdict vs pre-registration: prediction 0.20 FAILED (0.2183). Criterion "proven wrong if OOF >= 0.216": MET, so the hypothesis that dropout improves OOF is rejected at that threshold. Criterion "ratio stays >= 2x": not met (1.76). Dropout narrowed the gap mostly by RAISING train error (0.099 -> 0.124), not by lowering val error.
- exp05 vs exp04: +0.0021, below the 0.005 noise GUESS, so the head-to-head is INCONCLUSIVE. Seed noise is still unmeasured.
- Decision: by the pre-stated rule (A only if exp05 < exp04), Plan B. The AI recommends B (awaiting my choice).

## Audit 1 (2026-11-05, CPU only, while exp05 ran). Script md5 1b9fc6c3..., 116 lines
| check | evidence | result |
|---|---|---|
| rows -> breaths | L49 sorts by (breath_id,time_step); L57/60 reshape(-1,80). Printed: every reshaped row has one breath_id, time strictly increasing | PASS |
| folds grouped by breath | L83-87 give one fold label per breath. 67,905 breaths, pairwise overlap between folds = 0 | PASS |
| (R,C) balanced | crosstab of 9 RC groups x 5 folds: max-min per group = 1 breath. Fold sizes 13,577-13,584 | PASS |
| OOF on u_out==0 only | L106 masks with Mtr (u_out==0). Recomputed from saved fold0_oof.npy: exp02 0.24189, exp04 0.21621 (matches log). All steps would be 8.39/8.66 | PASS |
| loss denominator | L103 err[m].mean() = sum / number of scored steps. Toy check: 2.667 = sum/num_scored, not sum/all (1.0) | PASS |
| train vs val gap real? | Logged train (L105) is a running average in train mode while weights change, weighted by breaths, so it is not like-for-like. Recomputed exp04 best ckpt in eval mode, scored steps only: TRAIN 0.0991 (5,000 train-fold breaths) vs VAL 0.2162 | gap is REAL (2.2x) |
| scalers | L55 fit on ALL train rows incl. the validation fold (inputs only, no labels); never on test. Effect: max mean shift 0.0015 sd, max sd ratio 1.0% | minor leak of input stats, negligible; fixed in v2 |

Other audit notes:
- Best epoch is picked on the same fold it is scored on (L108-109), so OOF is slightly optimistic. LB so far is slightly BETTER than OOF, so no visible optimism.
- Predictions on u_out==1 steps are meaningless (MAE ~13): not trained and not scored. Harmless.

## Script versions
- v1 src/train_lstm.py (md5 1b9fc6c3...): exp01-exp05. Scalers fit on all train rows incl. the validation fold (Audit 1).
- v2 src/train_lstm_v2.py, first version (md5 1cb0980d...): ONLY change is per-fold scalers fit on the training folds, saved as fold{k}_scaler.npy. Folds come from src/common.py (checked identical to v1 folds).
- v2 bookkeeping update (md5 290e9fa4..., built from 1cb0980d by src/v2edit.py): --wd flag (default 1e-5 = old hard-coded value); fold{k}_args.json with script md5; fold{k}_test_best.npy saved at every validation improvement; --debug_breaths for smoke tests; CPU fallback. No change to training at default flags. Use this for all new experiments.
  - CPU smoke test (runs/smoke_v2, 256 breaths, 2 epochs, hidden 32): all files written; test_best == final test; submission ids in sample order; saved scaler mean equals the train-fold-only mean (max diff 3e-6) and NOT the train+val mean (diff 41). runs/smoke_v2 left in place (not deleted).
- src/overnight.sh (md5 486fb4fc...): Plan A/B queue, described below. bash -n ok; the pick rule was tested on existing logs (chooses the challenger only when it beats the baseline by more than the seed noise; falls back to the baseline on a missing log).
- src/ensemble.py: prints a per-fold OOF table (each run, MEAN, MEDIAN, +snap) and writes submission.csv with --agg mean|median (default mean). It uses only completed folds.
  - BUG FOUND IN TEST: the first version picked up the partial fold1_oof.npy from killed exp02 fold 1 (MAE 0.743). Fixed by requiring fold{k}_test.npy, which only exists when a fold finishes.
  - First result, exp02+exp04 on fold 0: MEAN 0.21251 (better than exp04 alone at 0.21621, even though exp02 is weaker); MEAN+snap 0.21126. With 2 runs mean == median by definition, so the median choice needs >= 3 runs (still untested).

## Overnight plan (12 h container, v2 script, all fold-0 unless noted; ~85 min per run = 300 x ~16 s + prep, GUESS for a new node)
Choose A if exp05's final fold-0 OOF < exp04's 0.21621, otherwise B. The decision rule for folds 1-4 is pre-registered in overnight.sh.
| plan | run | single change (vs) | hypothesis | expected fold-0 OOF (GUESS) | proven wrong if | GPU |
|---|---|---|---|---|---|---|
| A | exp06 | v2 scalers (vs exp05) | the scaler leak was negligible | exp05 +/- 0.005 | differs by > 0.005 (unless noise from exp08 is that big) | 85 min |
| A | exp07 | dropout 0.1 -> 0.2 (vs exp06) | more dropout narrows the gap further | exp06 - 0.005 | >= exp06 - noise | 85 min |
| A | exp08 | seed 42 -> 43 (vs exp06) | measures seed noise | exp06 +/- 0.005 | (measurement, not a test) | 85 min |
| A | best of exp06/07 folds 1-4 | fold (vs its fold 0) | gives a 5-fold model for the test ensemble | each fold within +/- 0.01 of fold 0 | any fold > 0.02 off (would suggest a fold problem) | 4 x 85 min |
| B | exp06 | v2 scalers (vs exp04) | the scaler leak was negligible | 0.2162 +/- 0.005 | differs by > 0.005 | 85 min |
| B | exp07 | weight decay 1e-5 -> 1e-2 (vs exp06; optimizer is AdamW, v1 L92 / v2 L106) | decoupled weight decay narrows the gap where dropout did not | 0.205-0.215 | >= exp06 - noise | 85 min |
| B | exp08 | seed 42 -> 43 (vs exp06) | measures seed noise | exp06 +/- 0.005 | (measurement) | 85 min |
| B | best of exp06/07 folds 1-4 | fold | 5-fold model | +/- 0.01 of fold 0 | any fold > 0.02 off | 4 x 85 min |
Total per plan: 7 x 85 min = ~9.9 h of an 11 h budget.
- Weight-decay sanity (computed): AdamW decouples decay (w -= lr_t*wd*w per step). Under our OneCycle schedule, sum of lr over 32,100 steps = 32.16 (mean lr 1.0e-3). The pure-decay shrink over training is exp(-wd * 32.16): wd 1e-5 -> 0.9997 (effectively none, as in all runs so far), 1e-3 -> 0.968, 1e-2 -> 0.725, 1e-1 -> 0.040. 1e-2 is PyTorch's AdamW default and a moderate setting. With plain Adam, wd is an L2 term divided by the adaptive denominator, so its strength would depend on gradient scale and 1e-2 could act very differently. That is not our case. Whether 1e-2 HELPS is a guess.
- Reaching the -b pod: the AI can only use web pages in your Chrome. It cannot ssh or kubesh. It reads the overnight results from the shared home folder through a DataHub Jupyter server after the pod is gone (one GPU per user). Evidence the home is shared: this server's /home/m9do is its-dsmlp-fs05:/export/workspaces/DSC102_FA26_001/home/m9do, and the pod is launched with -W DSC102_FA26_001. Not yet verified from inside the pod: check df -h ~ there.

## AI decisions / mistakes caught
- Features were chosen from physics: pressure is about R*flow + volume/C, so the AI added vol = cumsum(u_in*dt), vol/C, u_in*R, lags/leads. (AI design choice; check by ablation.)
- Loss is masked to u_out==0 because that is what is scored (from the metric definition, not a competition trick).
- MISTAKE: the AI assumed a bigger batch would fix the slow epochs. Measuring GPU utilization (97%) showed the model itself is the bottleneck, so the bigger batch helped only ~10%.
- MISTAKE: the AI picked 120 epochs to fit 5 folds in the 6 h container, without checking convergence. The exp02 curve was still improving at the end. Lesson: read the learning curve before scaling to more folds.
- MISTAKE: the AI picked hidden=192 to get a "smaller, faster" model. It was 2.1x slower per step than 256 (cuDNN slow path for a size that isn't a power of two). Caught because the epoch time went UP. A microbenchmark (ms per fwd+bwd step, bs512): 256x4 254, 192x3 532, 256x3 184, 128x4 121, 128x3 90. Lesson: benchmark instead of assuming.
- MISTAKE: the AI killed exp02 folds and submitted without a pre-stated hypothesis. Fixed by the ground rules above (approved by me in chat, but undocumented at the time).
- MISTAKE: the AI's stage table mixed data-backed numbers with guessed gains, unlabeled.
- MISTAKE (found by the AI in its own test run): ensemble.py v1 had no guard against partial OOF files from killed runs and reported fold 1 at 0.743. The test was run because I asked for the mean/median OOF change. (Test predictions were never affected: they only read fold*_test.npy, which a killed fold never writes.)
- MISTAKE (caught by me): the exp05 mid-run summary misstated the numbers in its own table. It said exp05 "led early (ep25-75)" (it was behind at ep75) and "trailed by ~0.009-0.010 since ep150" (actual gaps 0.036, 0.015, 0.013, 0.010, 0.008). The table values were correct (verified against the raw stdout lines); the prose was written from a partial printout, not the data. Lesson: derive summary sentences from the printed numbers, not from memory of them.
- MISTAKE (found by the AI when it re-read the log to verify an update): the log-update helper read EXPERIMENT_LOG.md through the browser HTTP cache, so updates were applied to stale copies and overwrote earlier sections. Undetected for several saves because the AI checked only the HTTP status (200), not the content. Fix: uncached reads, plus a re-read check after every save. Lesson: verify the written content, not the status code.
