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
- Decision: by the pre-stated rule (A only if exp05 < exp04), Plan B. The AI recommends B.
- 2026-11-06 ~00:40 UTC: Plan B APPROVED by me. Pre-launch check: overnight.sh B runs exp06 first; its args parsed with v2's argparse equal runs/exp04/args.json on every key (epochs 300, bs 512, lr 2e-3, hidden 128, layers 4, loss_mask insp, seed 42, nfolds 5, folds 0). The only extra keys are new v2 flags at v1-equivalent values: dropout 0.0, wd 1e-5, debug_breaths 0. Only intended difference: per-fold scalers (v2). Nothing running on the DataHub web server before shutdown (no train/queue processes, GPU 0%).

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


## Overnight Plan B results (pod m9do-2812086; read 2026-11-06 ~19:15 UTC from a CPU-only DataHub server)
Run integrity (from runs/overnight_B.txt and the files): queue start 01:38:09 UTC, done 10:32:36 UTC. 7/7 runs have START and END lines; each has 300/300 epochs, script md5 290e9fa4, the planned args (wd, seed), all 6 per-fold files, a test.npy with 603,600 finite rows equal to test_best.npy, and no Traceback/Error/nan/Killed in stdout. Nothing under runs/ was written after 10:32 UTC. Median 14.9-15.1 s/epoch, ~76-77 min per run.
Eval-mode train MAE: new src/eval_train.py (CPU; best checkpoint; model.eval(); u_out==0 steps only; the fold's own saved scaler; Net and feats taken verbatim from train_lstm_v2.py via ast, md5-checked). Its recomputed val MAE equals the logged best and the saved OOF file to 1e-6 for all 7 models (audit pass).

| run | fold | single change | OOF MAE (best ep) | eval train MAE | val/train | pre-registered | verdict |
|---|---|---|---|---|---|---|---|
| exp06 | 0 | v2 scalers (vs exp04) | 0.21491 (290) | 0.09833 | 2.19 | 0.2162 +/- 0.005; wrong if off by > 0.005 | consistent (-0.0013 vs exp04). Caveat: seed noise (0.0088) > 0.005, so this test could not detect a leak effect smaller than the noise. |
| exp07 | 0 | wd 1e-5 -> 1e-2 (vs exp06) | 0.21079 (273) | 0.10339 | 2.04 | 0.205-0.215; wrong if >= exp06 - noise = 0.20612 | falsification criterion MET (0.21079 >= 0.20612): not shown to help. It lands inside the expected range and is -0.0041 vs exp06, but that is under one seed-noise unit. Gap narrowed only slightly (2.19 -> 2.04: train +0.0051, val -0.0041). |
| exp08 | 0 | seed 42 -> 43 (vs exp06) | 0.22370 (287) | 0.09611 | 2.33 | exp06 +/- 0.005 (measurement) | outside my guessed band (+0.0088): the 0.005 noise guess was too small. |
| exp06 | 1-4 | fold | 0.21514 / 0.21981 / 0.21442 / 0.21492 | 0.09647 / 0.09753 / 0.10006 / 0.09964 | 2.23 / 2.25 / 2.14 / 2.16 | each within +/- 0.01 of fold 0; wrong if any > 0.02 off | PASS: largest deviation +0.0049 (fold 2). 5 folds: mean 0.21584, SD 0.00223, range 0.00539. |

Pick rule (pre-registered, applied by overnight.sh): base 0.21491, chal 0.21079, seed 0.22370, noise 0.00879, threshold 0.20612 -> exp06 for folds 1-4.

Seed noise, how far to trust it: one pair is one draw. sigma_hat = 0.0088/sqrt(2) = 0.0062; the 95% chi-square interval with 1 df is [0.0028, 0.198], so it is nearly uninformative. Cross-check (a different quantity): exp06's fold-to-fold SD is 0.0022, but all folds share seed 42, so it probably understates seed noise. If seed SD were 0.0022, a 0.0088 gap would have p ~ 0.005 (computed). So seed variance is probably larger than the fold spread (an inference, not measured). Consequence: single-run fold-0 comparisons cannot resolve effects < ~0.01. Future one-change tests need >= 2 seeds per arm.

Fold-0 ensembles (same breaths, all models trained without fold 0; computed in the kernel):
| models | n | mean | mean+snap | median | median+snap |
|---|---|---|---|---|---|
| 02+04 (current LB best, 0.20797) | 2 | 0.21251 | 0.21126 | - | - |
| 06 alone | 1 | 0.21491 | 0.21369 | - | - |
| 06+08 (same config, 2 seeds) | 2 | 0.20384 | 0.20248 | - | - |
| 04+06 | 2 | 0.20160 | 0.20025 | - | - |
| 06+07+08 | 3 | 0.19667 | 0.19524 | 0.19692 | 0.19552 |
| 04+06+07 | 3 | 0.19552 | 0.19411 | 0.19638 | 0.19502 |
| 04+06+07+08 | 4 | 0.19409 | 0.19266 | 0.19304 | 0.19158 |
| 02+04+06+07+08 | 5 | 0.19598 | 0.19450 | 0.19483 | 0.19340 |
- Data: averaging models is the largest gain measured so far (2 seeds of one config: -0.011 vs the better single; 4 models: about -0.020). The models have high variance; this matches the 2.2x train/val gap.
- Data: mean vs median is mixed (mean better at n=3, median better at n=4 and 5). One fold, so not settled.
- Data: snap gives -0.0012 to -0.0014 on every row and every fold [COMPETITION-SPECIFIC?].
- Caveat: choosing the best subset on fold 0 and scoring it on fold 0 is optimistic.

5-fold test ensemble of exp06 (NOT submitted): submissions/sub05_exp06_5fold_mean.csv and sub06_exp06_5fold_mean_snap.csv (submission.csv now = sub06). Also submissions/cand_all9_mean.csv (all 9 test files incl. exp02; candidate only). Checks for sub05/sub06: 603,600 rows; header id,pressure; ids equal sample_submission in order; 0 NaN; all finite; sub06 100% on the pressure grid; sub05 equals the mean of the 5 fold test files.
- A 5-fold test ensemble cannot be OOF-scored (models for folds 1-4 trained on fold-0 breaths). Fair fold-0 comparison: exp06 fold-0 alone 0.21369 (snap) vs current best 02+04 0.21126 (mean+snap): the single model is 0.0024 worse. Averaging proxy on the same breaths: 06+08 0.20248, 06+07+08 0.19524 (mean+snap).
- Data: the fold models disagree on test (mean |fold0 - fold k| on u_out==0 test steps 0.169-0.177), so averaging should help.
- GUESS: LB for sub06 is 0.195-0.205.

GPU quota: at 19:13 UTC a DataHub spawn with a GPU was refused ("GPU quota exceeded ... 1 already in use"). So a GPU pod (probably m9do-2812086) still holds the user's only GPU, 8.7 h after the queue finished. Not touched; deleting it is the user's call.
Tool note: ensemble.py treats the value after --agg as a run name (harmless, since no such run dir exists). Fix in its next version.

- 2026-11-06: my seed-noise guess (0.005, used in the exp06/exp08 pre-registrations) was too small; one measured pair gives 0.0088.
- 2026-11-06: my first DataHub spawn request went out with the GPU profile instead of CPU-only (the radio choice I set in JS did not take before the form submitted). The quota refused it, so there was no effect; I retried with a verified CPU-only choice.


## 2026-11-06 day session (DataHub GPU server, user approved: submit sub06 and sub07, run the throughput test and exp09)
Leaderboard (both submissions approved by the user in chat):
| sub | file | models | fold-0 proxy (OOF) | LB | commit / time (UTC) |
|---|---|---|---|---|---|
| sub06 | sub06_exp06_5fold_mean_snap.csv | exp06 folds 0-4, mean+snap | none (5-fold not OOF-scoreable); exp06 fold-0 alone 0.21369 | 0.18383 | b2e1825, 19:36:17 |
| sub07 | sub07_all8_mean_snap.csv | exp04 f0 + exp06 f0-4 + exp07 f0 + exp08 f0, mean+snap (exp06 has 5/8 weight) | none; the 4 fold-0 models together score 0.19266 on fold 0 | 0.18125 | 6aac2a2, 19:37:31 |
- sub06 beat my guess band (0.195-0.205) by about 0.011-0.021: the guess was too pessimistic. sub06 vs the previous best (0.20797): -0.0241.
- sub07 vs sub06: -0.0026 LB. NOTE: the sub07 mix (which models, equal weights per file) was chosen by the user's rule "every finished model", NOT on validation. No OOF number exists for it.
- sub07 checks: 603,600 rows; header id,pressure; ids equal sample_submission in order; 0 NaN; all finite; all on the grid; equals snap(mean of the 8 test files). Mean |sub07 - sub06| on u_out==0 test steps: 0.0353.

### Throughput test (3a): pre-registration, written before results
- Question: for hidden 128 x 4 (not hidden 256, where the "bigger batch does not help" conclusion came from), how do time/epoch and GPU utilization depend on batch size?
- One change: batch size 128 / 256 / 512 with LR scaled linearly (5e-4 / 1e-3 / 2e-3). Extra labelled rows: the same with the training tensors kept on the GPU (gpu_data True) instead of indexed on the CPU and copied each step as v2 does. Script src/bench_throughput.py: same step code as v2, fold-0 train breaths, 2 epochs per setting, nvidia-smi sampled every 0.5 s. Train time only (v2's logged sec also includes val prediction).
- Expected (GUESS): about 38 / 22 / 15 s per epoch; GPU utilization >= 90% at all sizes.
- Proven wrong if: bs 128 or 256 epoch time <= 1.1x bs 512's (batch size does not matter for speed).
- GPU time: ~10-15 min. It measures speed only; accuracy at each batch size is NOT tested here.

### exp09: pre-registration
- Purpose: a measurement, not a test. A third seed of the exp06 config (fold 0) for the noise estimate, plus one more ensemble member.
- One change vs exp06: seed 42 -> 44. Config: --epochs 300 --bs 512 --lr 2e-3 --hidden 128 --layers 4 (exactly exp06; not changed by the throughput result).
- Expected (GUESS): 0.215-0.225 (the two seeds so far: 0.21491, 0.22370).
- Would show the noise is even bigger than thought if it falls outside 0.205-0.235.
- GPU: ~77 min. Started after the throughput test finishes.


### Throughput test (3a): result (runs/bench_throughput.csv; train time only, fold-0 train breaths, 2nd epoch)
| batch | LR | steps/epoch | s/epoch (CPU-indexed, as v2) | s/epoch (data on GPU) | ms/step | GPU util median | peak mem |
|---|---|---|---|---|---|---|---|
| 128 | 5e-4 | 425 | 14.70 | 14.43 | 34.6 | 92% | 0.56 GB |
| 256 | 1e-3 | 213 | 14.08 | 13.92 | 66.1 | 96% | 1.01 GB |
| 512 | 2e-3 | 107 | 13.81 | 13.60 | 129.0 | 98% | 1.90 GB |
- Verdict: pre-registration PROVEN WRONG. bs 128 is 14.70 s, which is <= 1.1 x bs 512 (15.2 s). My guesses (38 / 22 s) were far off. For hidden 128 x 4 the GPU is already about 92% busy at bs 128, and time per epoch is nearly independent of batch size (+6% from 512 to 128).
- Consequence (data): bs 128 gives 4x more optimizer steps per epoch for about 6% more time. Keeping data on the GPU saves only about 2%; not worth a code change.
- Open question (not tested): whether bs 128 reaches the same OOF in fewer epochs. That is an accuracy question.

## CPU error analysis, fold 0 (src/error_analysis.py -> runs/analysis/; S = exp06 single, E = exp04+06+07+08 mean)
- By (R,C) (step-weighted MAE): R=50 is 40% of breaths but 51% of the single model's error (R50C50 0.322, R50C20 0.297, R50C10 0.229). Highest val/train ratios: R50C50 2.72, R5C10 2.62, R50C10 2.48; lowest: R5C50 1.48. Averaging gain is largest for R=50 (0.023-0.033).
- By position in inspiration: error rises from 0.158 (k = 0) to 0.236 (k = 15-18), dips to 0.219 (k = 27), then 0.265 at k = 30 (only 6,247 breaths reach k = 30).
- Concentration: breath MAE median 0.127, p90 0.405, p99 1.49, max 12.0 (single). The worst 1/5/10% of breaths hold 12/29/41% of the error (single), 12/31/43% (ensemble).
- Averaging gain by difficulty decile: none on the easiest 10% (-0.0015), +0.10 absolute on the hardest 10%. Relative gain about 10% from decile 5 up.
- Worst 20 breaths (runs/analysis/worst20.png, worst20.csv): 18/20 are R = 50 with LOW u_in (mean during inspiration 4-12) and plateau pressures 20-42. The other 2 are R5C10 outliers; one has u_in hitting 100 (MAE 7.3), one has MAE 11.5.
- From the plots: in those 18, the model gets the SHAPE of the inspiratory pressure right (the oscillations line up) but the LEVEL is offset by a near-constant 3-5 cmH2O, in both directions (9 too high, 11 too low).
- Level vs shape (computed, all fold-0 breaths): removing each breath's mean signed error on scored steps would cut S from 0.21491 to 0.13895 and E from 0.19409 to 0.11976. ORACLE CEILING: this assumes the true per-breath offset is known, which it never is at test time; it is NOT an achievable result, only an upper bound on what fixing the level could give. So 35% (S) / 38% (E) of all error is a per-breath constant offset. For the worst 20: 4.31 -> 1.39 (same oracle caveat). Similar share for every R (0.38-0.39). Mean |offset| is largest at R = 50 (0.18 vs 0.11-0.13).
- The offset is systematic, not variance: the per-breath offsets of S and E correlate 0.91. That is why averaging does not fix them.
- Is the offset predictable from simple breath summaries? (R, C, inspiration length, u_in insp mean/max/std/first/last, u_in expiration mean/last, dt mean/max, end time; HistGradientBoosting, 5-fold CV within fold-0 val.) R^2 = 0.009. Subtracting the predicted offset makes OOF worse (0.19564 vs 0.19409). Answer: no. Every |Spearman| with |offset| is <= 0.12.
- Rarity: nearest training-fold breath (same R,C) by Euclidean distance on u_in over the first 32 steps. Spearman(NN distance, breath MAE) = 0.047. The worst 20 sit at median percentile 0.53 and the worst 1% at 0.24; only 2/20 are in the rarest decile (the two R5C10 outliers). The closest-neighbour decile has the HIGHEST MAE (0.28).
  - Caveat: this distance is scale-confounded (low-u_in breaths have small distances, and low-u_in R=50 breaths are the hard ones), so it is a weak test.
  - Conclusion: "the worst breaths are rare u_in patterns" is NOT supported, except for the 2 outliers.
- Interpretation (GUESS, labelled): the hard breaths need a breath-level quantity (the plateau level) that our models do not extract from the inputs. Simple summaries do not contain it either. If it is recoverable at all, it is in the detailed u_in dynamics. [COMPETITION-SPECIFIC?] The pressure oscillations mirror u_in, which suggests u_in comes from a feedback controller reacting to pressure. If so, u_in's fine detail carries pressure information. This is my reading of our own plots, not from any outside source.

## Proposed queue for tonight (NOT approved yet; nothing started)
Controls already exist: exp06 (seed 42) 0.21491, exp08 (seed 43) 0.22370, exp09 (seed 44) running. All are the exp06 config, fold 0.
Decision rule for every arm (pre-registered): pooled seed SD s from the 3 control seeds; adopt a challenger only if mean(challenger, 2 seeds) < mean(controls) - 2 * s * sqrt(1/2 + 1/3). GUESS: s ~ 0.005-0.006, so the detectable effect is ~0.009-0.011.
| arm | runs | single change vs exp06 | hypothesis | expected fold-0 OOF, mean of 2 (GUESS) | proven wrong if | GPU |
|---|---|---|---|---|---|---|
| T1 | exp10a/b (seeds 42, 43) | --loss_mask all (supervise all 80 steps) | expiratory pressure carries information about the same breath-level latent that sets the inspiratory level, so 2.5x more targets per breath reduce the level offset | 0.205-0.215 | rule not met, OR the oracle level share (35%) does not drop | 2 x 77 min |
| T2 | exp11a/b (seeds 42, 43) | --epochs 150 | half the schedule gets within noise of 300 | controls + 0.000 to +0.008 | mean(150) > mean(controls) + 2 * s * sqrt(1/2 + 1/3) | 2 x 39 min |
| T3 | exp12a/b (seeds 42, 43) | --hidden 256 | the offset is shared by every model we trained (corr 0.91), i.e. a bias of this model size; more capacity can decode finer u_in detail | 0.200-0.212 | rule not met | 2 x ~160 min (254 vs 121 ms/step measured earlier for 256x4 vs 128x4) |
Total about 9.2 h. If T2 passes, every later experiment costs half.


## Day queue on the DataHub GPU server (approved 2026-11-06; T3 on hold)
Server stop time: the container started at 19:35:47 UTC (measured: start time of PID 1). No expiry is exposed (container env has no timeout variable; the hub API from inside the container returns no expiry). The 6 h limit is from the handout (dsmlp-login launch) and has NOT been observed for a web-launched server (yesterday's was stopped by hand). Planning assumption: stop at 01:35 UTC = 5:35 PM PT. Possible extra risk (GUESS): JupyterHub idle culling if the browser tab closes; background processes may not count as activity.
Run names (changed from the proposal, now in run order): T2 = exp10a/b, T1 = exp11a/b.
Queue: src/dayqueue.sh (launched 20:20:41 UTC; waits for exp09) -> exp10a, exp10b -> src/decide.py T2 -> exp11a, exp11b at the epochs T2 chose -> src/decide.py T1. Decisions go to runs/decision_T2.json / decision_T1.json and runs/dayqueue.txt. The decide.py logic was dry-run on stand-in runs in a temp dir: pass, fail and missing-run cases behave as written.
Timing (estimate from 15.3 s/epoch + ~1.5 min prep): exp09 ends ~20:57; T2 ends ~22:17; T1 ends ~23:37 if 150 epochs, ~00:55 if 300. Both are before the assumed 01:35 stop.
| arm | runs | single change | reference | expected fold-0 OOF, mean of 2 (GUESS) | rule (pre-registered) |
|---|---|---|---|---|---|
| T2 | exp10a/b, seeds 42/43 | --epochs 300 -> 150 | controls exp06/08/09 (300 ep, seeds 42/43/44) | controls + 0.000 to +0.008 | PASS if mean(T2) <= mean(ctrl) + 2*s_ctrl*sqrt(1/2+1/3), s_ctrl = SD(ddof 1) of the 3 controls. If PASS, every later run uses 150 epochs. |
| T1 | exp11a/b, seeds 42/43 | --loss_mask insp -> all | if 150 ep: exp10a/b (same seeds, same epochs); if 300: controls | reference - 0.005 (range -0.012 to +0.002) | ADOPT if mean(T1) < mean(ref) - 2*SE. SE = s*sqrt(1/2+1/2) with s pooled from controls (2 df) and the T2 pair (1 df) at 150; SE = s_ctrl*sqrt(1/2+1/3) at 300. Secondary (from the proposal): the oracle level share (35% for exp06) should drop. |
Power caveat (computed on the dry run, s ~ 0.0066): T2 passes unless 150 epochs is worse by ~0.012, so a PASS means "not detectably worse", not "equivalent". T1 can only detect gains of ~0.011-0.013. Both tests are weak with 2 seeds; that is the price of single-day budgets.
T3 (hidden 256): HELD by the user. Evidence against (from the user): exp02 (hidden 256, 120 epochs) 0.24189 vs exp04 (hidden 128, 300 epochs) 0.21621. That comparison is confounded by epochs (exp02 was still improving at epoch 120).
- 2026-11-06 PLANNING MISTAKE (AI): I proposed a 9-hour queue for "tonight" while this server's GPU would sit idle all afternoon (exp09 ends ~1 PM PT). Caught by the user. Lesson: schedule the next decisive tests on whatever GPU is free now; keep the overnight pod for what can't fit.


## Train-fold level diagnostic (CPU; runs/analysis/train_offset_diag.py; exp06 fold-0 model on its own training breaths, eval mode)
| | MAE | shape-only (ORACLE CEILING) | level share | mean abs offset |
|---|---|---|---|---|
| train breaths (54,321) | 0.09833 | 0.08776 | 0.108 | 0.044 (median 0.025, p95 0.13) |
| val breaths (13,584) | 0.21491 | 0.13895 | 0.353 | ~0.14 (median 0.066, p95 0.615) |
- Level share on train by R: R5 0.157, R20 0.129, R50 0.065. On val it is ~0.38 for every R.
- Data: the model places the level almost exactly on breaths it was trained on, but not on unseen breaths. Of the val-train gap (0.1166), 0.0654 (56%) is level error, so the level is the main thing that does not generalize.
- Interpretation (GUESS): the network memorises each training breath's level instead of learning a rule that carries over. Consistent with: the offset is not predictable from simple summaries (R^2 0.009), dropout and wd 1e-2 barely moved the gap, and exp02 (hidden 256) was worse.

### T3 (hidden 256): recommendation DROP for now
- The user's point stands: exp02 (hidden 256) 0.242 vs exp04 (hidden 128) 0.216, though confounded by 120 vs 300 epochs.
- New evidence: the failure is generalization of a memorised breath-level quantity, not underfitting (train MAE 0.098, train level share 11%). More capacity mainly adds memorization room.
- I cannot argue T3 is worth ~5 GPU-hours (2 seeds x ~2.5 h).

### Proposed instead (NOT run; awaiting approval): L1, a CPU level-corrector test, 0 GPU
- Question: is the per-breath offset recoverable from the full input sequence by another learner? If not, stop chasing it with model changes.
- Hypothesis: part of exp06's held-out offset is predictable from the full u_in sequence, the u_out pattern, R, C, and the model's own predicted inspiratory pressures.
- One change: subtract a predicted per-breath offset from exp06's predictions (model unchanged).
- Data: exp06 OOF from all 5 folds (67,905 breaths). The corrector for fold k is trained only on the other 4 folds' OOF residuals, then applied to fold k (clean CV, no test data). Learner: HistGradientBoosting.
- Expected (GUESS): CV R^2 of the offset 0.00-0.10; 5-fold OOF 0.21584 -> 0.211-0.216.
- Proven wrong if: R^2 < 0.02, or the OOF does not improve by >= 0.002 on at least 4 of 5 folds (paired comparison on the same predictions, so seed noise does not apply).
- CPU ~10 min. Not competition-specific: post-hoc residual stacking.

## Evening queue (12 h pod): DRAFT, decided after T2 (and T1 if it is done in time)
Principle: every arm targets breath-level memorization, has one change and 2 seeds (42/43), and is compared with the day-queue reference at the same epoch count.
| arm | single change | hypothesis | GPU (150 ep / 300 ep) | needs code? |
|---|---|---|---|---|
| N1 | bs 512 -> 128, LR 2e-3 -> 5e-4 (scaled) | 4x more, noisier updates per epoch regularize; same wall time (measured 14.7 vs 13.8 s/epoch) | 2 x 41 / 2 x 80 min | no |
| N2 | wd 1e-5 -> 1e-2, 2 seeds | re-test exp07's -0.0041 hint properly; decoupled wd penalizes the large weights that memorization needs | 2 x 40 / 2 x 78 | no |
| N3 | control seed 44 at the day queue's epoch count | firms up the noise estimate for that epoch count | 1 x 40 / 1 x 78 | no |
| N4 | input-noise augmentation on raw u_in during training (features recomputed) | makes exact u_in sequences unmemorisable, forcing a rule | 2 x ~45 / 2 x ~85 | yes (v3; CPU smoke test before launch) |
| N5 | if T1 adopted: T1 + N-winner combination; else a 3rd seed of T1 | - | 2 x 40 | no |
At 150 epochs N1-N5 is about 6.5 h, leaving room for N4 at 2 more seeds or folds of the best config. At 300 epochs, drop N5 and N4's second seed.


### L1 result (approved by the user, run 20:27 UTC on CPU; src/L1_corrector.py -> runs/analysis/L1_corrector.csv)
Setup as pre-registered: 121 features (u_in x 80, exp06's predicted pressure on steps 0-31, R, C, inspiration length, dt mean/max, end time, predicted insp mean/max, u_out count). Corrector for fold k trained only on the other 4 folds' OOF residuals. HistGradientBoosting with max_iter 500, lr 0.05, 63 leaves, early stopping; all fixed before any result.
| variant | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | pooled OOF (base 0.21584) | CV R^2 of offset |
|---|---|---|---|---|---|---|---|
| PRIMARY: squared-error loss (pre-registered) | -0.00139 | -0.00227 | +0.00009 | +0.00033 | -0.00136 | 0.21676 (worse by 0.00092) | 0.061 |
| secondary, NOT pre-registered: absolute-error loss | +0.00065 | +0.00076 | +0.00054 | +0.00053 | +0.00022 | 0.21530 (better by 0.00054) | 0.019 |
(fold cells = OOF gain; positive = better)
- Verdict: PROVEN WRONG. R^2 0.061 clears the 0.02 bar, but the OOF criterion fails: 0/5 folds improved by >= 0.002, and the primary corrector made 3/5 folds worse.
- The absolute-error variant improves every fold, but by only 0.0002-0.0008, a quarter of the bar. Its R^2 is 0.019. It was added after the pre-registration, so it is reported, not adopted.
- Interpretation (GUESS): the squared-error corrector explains some variance in the large offsets but moves typical breaths the wrong way under MAE. Neither variant recovers a useful part of the per-breath level from these inputs.
- Consistent with the train-fold diagnostic: the level is placed by memorisation on seen breaths and is mostly not inferable for new ones, at least by a tree model on these features.
- Not ruled out: a sequence learner could still extract it; T1 (all-step supervision, running today) is one test of that.
- Consequence for the evening draft: deprioritize further level-correction ideas. Keep the arms that attack memorization directly (N1 batch 128, N2 wd 1e-2, N4 u_in noise).


### exp09 result (seed 44, exp06 config, 300 epochs)
- Best val 0.22432; the log was last written 20:58:54 UTC and the queue logged "exp09 finished" at 20:59.
- Inside its pre-registered 0.215-0.225, so the noise is not even bigger than thought.
- Timing check: I read epoch 184 at 20:28 UTC; 116 x 15.3 s predicts 20:57:40. Actual 20:58:54 (mean 15.46 s/epoch over the last 30). T2 (exp10a) started 20:59 UTC.
- Controls now (300 epochs, seeds 42/43/44): 0.21491 / 0.22370 / 0.22432, mean 0.22098, SD 0.00526 (2 df). T2 pass threshold = 0.23059.
- Side note: exp07 (wd 1e-2, 1 seed) vs the 3-control mean is -0.01019, but 2 SE for 1-vs-3 is 0.01215, so still not significant.
- CORRECTED (AI): I first logged that my CPU jobs had slowed exp10a to 16.2 s/epoch. Wrong. exp10a's median over its first 20 epochs is 15.43 s (exp09 15.29, exp06 15.0 over their first 25). The 16.2-16.3 s epochs are the ones where val improved (v2 then also predicts the test set). I mis-attributed it from two log lines without checking the median. Lesson (same as before): compute from the logs, not from a glance.

### u_in noise (N4): label validity and level. Answer owed to the user.
- Before measuring, I had given NO noise level and no argument that the labels stay valid. Strictly they do not: pressure is the response to the real u_in, so any perturbation changes the true pressure.
- The only defensible version is noise small enough that the implied pressure change is below label resolution. Rule set before measuring: mean |dP| on scored steps <= 0.035 (half the 0.0703 grid step).
- Measured (runs/analysis/uin_noise_sensitivity.*): exp06 fold-0 model as a stand-in for the unknown physics; 4,000 fold-0 val breaths; iid Gaussian noise on u_in, clipped to [0,100], features recomputed. Scored-step u_in: median 7.6, mean 14.8, 14.7% exactly 0.
| sigma (u_in units) | mean abs dP | p95 abs dP | mean abs dP, R=50 | model MAE vs truth (base 0.2137) |
|---|---|---|---|---|
| 0.02 | 0.055 | 0.211 | 0.080 | 0.2306 |
| 0.05 | 0.113 | 0.486 | 0.174 | 0.2604 |
| 0.1 | 0.193 | 0.838 | 0.286 | 0.3131 |
| 0.25 | 0.343 | 1.437 | 0.499 | 0.4361 |
| 0.5 | 0.529 | 2.067 | 0.763 | 0.5944 |
| 1.0 | 0.776 | 2.956 | 1.116 | 0.8198 |
| 2.0 | 1.112 | 3.980 | 1.615 | 1.1425 |
- Verdict: no tested level meets the rule; even sigma 0.02 gives 0.055 > 0.035. N4 is DROPPED.
- Caveat: this is the MODEL's sensitivity, not physics. The true dP/du_in cannot be measured from observational data. The model may be over-sensitive.
- Finding in its own right (data): a 0.02-unit perturbation (~0.3% of median u_in) shifts predictions by 0.055 and raises MAE by 0.017. The network keys on tiny details of u_in. That fits the memorization picture.
- [COMPETITION-SPECIFIC?] It also fits the earlier guess that fine u_in detail carries pressure information.

## Evening queue: DRAFT v2 (src/evening.sh, NOT launched; finalize after T2's verdict)
Answers to the user's questions:
(2) Every challenger runs seeds 42 AND 43. A third seed (44) appears only for controls (N3) and, if T1 is adopted, for T1 (N5).
(3) N1 batch 128 keeps LR at 2e-3 (unchanged), so batch size is the only change. OneCycle total steps follow automatically (4x the steps). Scaling LR would be a second change.
(5) Current wd = 1e-5 (v2 L27 default; v1 L92 hard-coded). The optimizer is AdamW (v2 L106; v1 L92), so wd is decoupled.
GPU time per run: 300 epochs = 76-77 min (measured: overnight START/END stamps). 150 epochs = ~39 min (estimate; exp10a will measure it). bs 128 = +6.4% train time (measured in the throughput test): ~81 / ~41 min.
| arm | runs | single change vs reference | hypothesis | expected, mean of 2 (GUESS) | GPU at 150 ep | GPU at 300 ep |
|---|---|---|---|---|---|---|
| N1 | exp12a/b (42, 43) | bs 512 -> 128, LR 2e-3 unchanged | 4x more, noisier updates regularize against breath memorization | ref - 0.005 (range -0.015 to +0.010); instability at LR 2e-3 would also count against it | 2 x 41 = 82 min | 2 x 81 = 162 min |
| N2 | exp13a/b (42, 43) | wd 1e-5 -> 1e-2 | decoupled wd shrinks the large weights memorization needs (exp07 hint: -0.010 vs controls, n.s.) | ref - 0.006 | 2 x 39 = 78 min | 1 x 77 (exp07 is seed 42) |
| N3 | exp14 (44) | none (control) | third reference seed at 150 epochs | = ref | 39 min | 0 (exp09 exists) |
| N5 | exp11c (44) | T1, only if adopted | firms up T1 | - | 39 min | 77 min |
| total | | | | | 3.3-4.0 h | 4.0-5.3 h |
- References: at 150 epochs exp10a/exp10b/exp14 (seeds 42/43/44); at 300 epochs exp06/08/09.
- Rule (pre-registered): ADOPT if mean(challenger) < mean(reference) - 2*s*sqrt(1/2+1/3), with s = SD of the 3 reference seeds.
- Spare time on a 12 h pod (~11 h usable): about 7 h at 150 epochs, 5.7 h at 300.
- OPTION for the user (not included): a seed-44 run for N1 and N2 (+80 min at 150, +158 min at 300). It raises power (detectable effect drops from ~0.011 to ~0.009 if s ~ 0.005).
- Launch needs this server's GPU released first (1-GPU quota): the day queue ends ~23:40 UTC (150) or ~01:00 UTC (300).


## T2 verdict (decided by src/decide.py at 22:18 UTC)
| run | epochs | seed | best fold-0 OOF (epoch) |
|---|---|---|---|
| exp10a | 150 | 42 | 0.25689 (147) |
| exp10b | 150 | 43 | 0.25429 (147) |
| controls exp06 / exp08 / exp09 | 300 | 42 / 43 / 44 | 0.21491 / 0.22370 / 0.22432 (mean 0.22098, SD 0.00527) |
- T2 mean 0.25559; diff +0.03462; pass threshold 0.23059. FAILED decisively (expected +0.000 to +0.008: my guess was far off).
- Both 150-epoch runs still had their best at epoch 147 of 150, so they were still improving.
- Consequence: T1 runs at 300 epochs (exp11a started 22:18 UTC). 300 epochs stays the budget unit; no doubling of the experiment budget.
- Measured: exp10a/b ran 39-40 min each (START/END stamps).

## N4 reinstated (user decision) + MISTAKE
- MISTAKE (AI, caught by the user): I judged u_in noise with exp06's own sensitivity as a stand-in for the physics. That proxy assumes its own conclusion: over-sensitivity to tiny u_in changes is the memorization symptom N4 is meant to treat, so a memorizing model will always fail that bar. The fair judge is validation MAE on clean data: if the noise biases the labels, validation gets worse and the arm loses. The sensitivity table stays in the log as a diagnostic, not as a verdict.
- Chosen noise: MULTIPLICATIVE, u_in' = u_in * (1 + 0.01 * z), z ~ N(0,1), clipped to [0,100]; training batches only; every u_in-derived feature recomputed (vol, u_in_cum, lags/leads, d1, d2, vol_over_C, uin_x_R). Validation and test inputs stay clean.
- Why sigma = 1% (data, computed 23:07 UTC):
  - The relative step-to-step change of u_in on scored steps (u_in > 0.5) has median 8.1%, p25 3.1%, p10 1.1%. 1% is ~1/8 of the typical change, so the shape of the control signal survives.
  - Multiplicative noise leaves the 14.7% of scored steps with u_in exactly 0 at 0, where additive noise plus clipping would bias them upward. 0.9% of scored steps sit at 100 and get clipped.
- Code: src/train_lstm_v3.py (md5 4193f506) = v2 + --uin_noise; default 0 keeps the v2 code path.
- CPU smoke tests (CUDA hidden):
  - (1) Feature parity, 500 breaths: no-noise recompute vs feats() max diff 1.4e-6 SD; noisy recompute vs feats() on an identically noised dataframe max diff 6.9e-7 SD; zeros stay zero; mean relative change 0.80%.
  - (2) Equivalence: v2 vs v3 with --uin_noise 0 (256 debug breaths, 2 epochs, seed 42): train and val logs bit-identical.
  - (3) v3 with --uin_noise 0.01: runs, finite, and differs at full precision from the no-noise run (epoch-1 train 16.858408 vs 16.858551), so the noise path is active. It writes oof/test_best/test files.
- N4 pre-registration:
  - Hypothesis: 1% training-time u_in noise makes exact sequences unmemorisable and forces a transferable rule.
  - Expected (GUESS): 2-seed mean = controls - 0.006 (range -0.015 to +0.010).
  - Proven wrong if: it fails the evening rule, which includes the case where the noise biases labels and validation gets worse.
  - GPU: ~81 min per run (GUESS: v2's 77 + ~5% for the per-batch recompute).

## L1 (restated for the record; full table above)
PROVEN WRONG. Primary corrector: pooled OOF 0.21584 -> 0.21676 (worse); fold gains -0.00139/-0.00227/+0.00009/+0.00033/-0.00136; CV R^2 0.061; 0/5 folds improved by >= 0.002. Secondary (not pre-registered): +0.0002 to +0.0008 per fold, R^2 0.019, not adopted. The offset was NOT found predictable, so nothing is cut for it.

## Evening queue: FINAL (src/evening.sh; verdicts by src/decide_evening.py)
300 epochs (T2 failed). Controls exp06/08/09 (seeds 42/43/44). Rule: ADOPT if mean(arm) < mean(ctrl) - 2*s*sqrt(1/n + 1/3), s = 0.00527.
| order | run | arm | change vs controls | seed | GPU (300 ep) |
|---|---|---|---|---|---|
| 1 | exp12a | N1 | bs 512 -> 128, LR 2e-3 unchanged | 42 | ~81 min (bs128 +6.4% measured) |
| 2 | exp12b | N1 | same | 43 | ~81 |
| 3 | exp13b | N2 | wd 1e-5 -> 1e-2 (AdamW) | 43 | 77 |
| 4 | exp15a | N4 | 1% multiplicative u_in noise (v3) | 42 | ~81 (GUESS) |
| 5 | exp15b | N4 | same | 43 | ~81 (GUESS) |
| 6 | exp12c | N1 | seed 44 (user: more power) | 44 | ~81 |
| 7 | exp13c | N2 | seed 44 | 44 | 77 |
| 8 | exp11c | N5 | T1 third seed, ONLY if T1 adopted | 44 | 77 |
- N2's seed 42 is exp07 (same script md5, same config).
- Detectable effect (2 SE): N1 and N2 with 3 seeds 0.0086; N4 with 2 seeds 0.0096.
- Total: 559 min (9.3 h) without N5, 636 min (10.6 h) with it, on a 12 h pod. Margin 1.4-2.7 h if the pod's GPU runs at the measured 15.0-15.4 s/epoch.
- The script refuses to start unless runs/DAYQUEUE_DONE exists. The day queue ends ~00:53 UTC (4:53 PM PT): exp11a ~23:35, exp11b ~00:53.
- No N3 needed at 300 epochs: exp09 is the third control.
