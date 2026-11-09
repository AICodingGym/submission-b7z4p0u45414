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


## sub08 (approved by the user; submitted 23:17:22 UTC, commit e252839)
- File: sub08_all9_mean_snap.csv = mean+snap of 9 test files: sub07's 8 (exp04 f0, exp06 f0-4, exp07 f0, exp08 f0) + exp09 f0.
- Checks: 603,600 rows; header id,pressure; ids equal sample_submission in order; 0 NaN; all finite; all on the grid; equals snap(mean of the 9 files). Mean |sub08 - sub07| on test u_out==0 steps: 0.0147.
- LB 0.18178 vs sub07 0.18125: +0.0005, i.e. no gain. The mix was not chosen on validation (same rule as sub07: every finished model, equal weight per file). exp09 is the weakest fold-0 single (0.22432).
- Current best stays sub07 (0.18125).

## Benchmark-agent inspection (handout Section 4, step 1): BLOCKED, no code reachable (2026-11-06 ~23:15 UTC)
What the challenge page shows (aicodinggym.com, "AI BENCHMARK, MLE-bench" panel; tooltip text, verbatim facts):
| agent | score | description (tooltip) | model(s) | time limit | hardware | link in tooltip |
|---|---|---|---|---|---|---|
| Disarray | 0.15171 | runs four frontier LLMs on each competition and combines their submissions | Claude-Opus-4.5, Claude-Sonnet-4.5, GPT-5.2-Codex, Gemini-3-Pro-Preview | 24 h | 24 vCPU, 220 GB, 1x A100 | disarray.ai |
| AIBuildAI | 1.0759 | Claude-Opus-4.6 autonomous loop: plan -> code -> train -> evaluate -> submit | Claude-Opus-4.6 | 24 h | 24 vCPU, 256 GB, 1x A100 | github.com/aibuildai/AI-Build-AI |
| MLEvolve | 0.14237 | Gemini-3-Pro-Preview iteratively proposing, running and refining ML pipelines | Gemini-3-Pro-Preview | 12 h | 21 vCPU, 234 GB, 1x H200 | github.com/InternScience/MLEvolve |
Where the "source" link leads: github.com/openai/mle-bench/tree/main/runs. It holds one folder per run group (disarray_group_1-3, mlevolve_group1-3; no AIBuildAI folder). Each folder holds only a grading_report JSON (Git LFS) with scores, NOT code.
Agent repos (from the tooltips), checked by file path via the GitHub API:
- InternScience/MLEvolve: 91 files, framework code only; no path contains "ventilat".
- aibuildai/AI-Build-AI: 1,079 files; no path contains "ventilat".
Result: none of the three agents' ventilator code (features, model, validation, loss mask, submission) is reachable from the links. Per the user's instruction I stopped and asked for the files instead of guessing. No claims about their methods are made until the code is attached.


## Benchmark agent source unavailable (logged Sat Nov 7, from notes made Fri night while DataHub was off)

"Benchmark agent source unavailable: the source link holds only grading reports; no ventilator code in the MLEvolve or AI-Build-AI repos."

The user confirmed MLE-bench publishes only grading reports, so the inspection is CLOSED. No agent files are coming.

Leaderboard tooltip table (all we know about the three agents):
| agent | LB MAE | what it is | budget | hardware | linked source |
|---|---|---|---|---|---|
| Disarray | 0.15171 | 4 LLMs combined: Opus 4.5, Sonnet 4.5, GPT-5.2-Codex, Gemini 3 Pro | 24 h | A100 | disarray.ai |
| AIBuildAI | 1.0759 | Claude Opus 4.6 plan-code-train-evaluate-submit loop | 24 h | A100 | github.com/aibuildai/AI-Build-AI |
| MLEvolve | 0.14237 | Gemini 3 Pro iterative pipelines | 12 h | H200 | github.com/InternScience/MLEvolve |
Source link on the leaderboard: github.com/openai/mle-bench/tree/main/runs (grading reports only; Git LFS pointers).

### Why did AIBuildAI score 1.08? Every item below is an UNTESTED GUESS (no code was available)
- Reference points from our own data: per-(R,C,step) median baseline 6.08; global median 7.54; our first 120-epoch LSTM 0.242. So 1.08 learned real signal but missed something big.
- G1 (UNTESTED GUESS): a per-row tabular model (e.g. GBM) with little or no within-breath sequence context. The error size is plausible.
- G2 (UNTESTED GUESS): training, early stopping or model selection on all 80 steps instead of u_out==0. Partial evidence AGAINST this being the whole story: our T1 (loss on all steps, exp11a/b) stayed near 0.22, not 1.0.
- G3 (UNTESTED GUESS): submission row-order / id misalignment in part of the file. A full misalignment would usually score much worse than 1.08, so it would have to be partial.
- G4 (UNTESTED GUESS): ran out of its time budget and submitted an early or fallback model.
- The leaderboard score alone cannot separate these.


## Sat Nov 7: evening-queue results (pod, 06:46-15:55 UTC) and fold-0 analysis
- Completion check (runs/evening.txt): all 7 runs started and printed "FOLD 0 DONE" (exp12a, exp12b, exp13b, exp15a, exp15b, exp12c, exp13c). exp11c was skipped as designed (T1 REJECT). EVENING_DONE was written at 15:55 UTC. Every run has fold0_oof / fold0_test / fold0_best.pt.
- runs/decision_evening.json on disk was a stale dry run (Nov 6 23:08). Re-ran src/decide_evening.py today; it overwrote the file.
- Controls: exp06/08/09 = 0.21491 / 0.22370 / 0.22432, mean 0.22098, s 0.00527. Rule: ADOPT if mean(arm) < mean(ctrl) - 2*s*sqrt(1/n + 1/3).

| arm | change | runs (fold-0 OOF) | n | mean | diff vs ctrl | threshold | pre-registered expectation | verdict |
|---|---|---|---|---|---|---|---|---|
| T1 | loss on all 80 steps | exp11a 0.22190, exp11b 0.21915 | 2 | 0.22052 | -0.00045 | 0.21136 | 0.205-0.215, and level share drops | REJECT. Proven wrong on both criteria: rule not met; level share 0.348 / 0.359 vs controls 0.353-0.361 (no drop) |
| N1 | bs 512 -> 128 (LR 2e-3 unchanged) | exp12a 0.20142, exp12b 0.20460, exp12c 0.20869 | 3 | 0.20490 | -0.01607 | 0.21238 | ref - 0.005 (range -0.015 to +0.010) | ADOPT. All 3 seeds beat the best control. Effect larger than my range |
| N2 | wd 1e-5 -> 1e-2 | exp07 0.21079, exp13b 0.21701, exp13c 0.21783 | 3 | 0.21521 | -0.00576 | 0.21238 | ref - 0.006 | NOT ADOPTED (point guess matched, but the rule is not met; could be noise) |
| N4 | 1% multiplicative u_in noise (v3) | exp15a 0.23915, exp15b 0.24536 | 2 | 0.24225 | +0.02128 | 0.21136 | ref - 0.006 | REJECT. Proven wrong: clearly worse |

### N1 mechanism: my hypothesis was WRONG even though the arm passed
- Pre-registered hypothesis: "4x more, noisier updates regularize against breath memorization".
- Eval-mode train MAE (src/eval_train.py, runs/eval_train.csv): exp12a/b/c 0.0572 / 0.0568 / 0.0591 vs controls exp06/08/09 0.0983 / 0.0961 / 0.0980. N2 exp13b/c 0.0983 / 0.1033.
- bs 128 fits the TRAIN fold far better, not worse. The train/val ratio rose from about 2.2 to 3.5. So it is not regularization.
- Better reading (inference): the bs-512 runs were optimization-limited. bs 128 gives 4x the optimizer steps, and both train and val improve. This agrees with T2 (half the epochs cost +0.035).

### Per-breath level offset after the evening runs (runs/analysis/evening_fold0.py, .txt)
- Level share = 1 - MAE(err - per-breath mean insp residual) / MAE(err). This is an ORACLE decomposition, not an achievable score.
- Controls (exp06/08/09), mean of single runs: mean |offset| 0.1630, shape-only MAE 0.1419, level share 0.357.
- N1 (exp12a/b/c), mean of single runs: mean |offset| 0.1527 (-6%), shape-only MAE 0.1297 (-9%), level share 0.367.
- Ensembles: N1 mean-of-3 OOF 0.18517, |offset| 0.1407, shape 0.1129, level share 0.391. S8 mean-of-5 0.19339, level share 0.386.
- Per-breath offset correlation: between N1 seeds 0.76-0.80; exp06 vs exp12a 0.73.
- Interpretation (inference): averaging models removes shape error faster than level error, so most of the level offset is SHARED across models (systematic), not seed noise. bs 128 shrank both parts; nothing so far targets the level specifically (L1 corrector, T1 and ensembling all failed to).

### Stacking
- Only N1 passed, so the stacked config is the exp06 config with bs 128. It is the NEW REFERENCE: 3 seeds, mean 0.20490, SD 0.00364. Pooled SD with the old controls: 0.00453.
- N2 is not stacked because it failed its rule.

### Ensemble check on fold 0 (MEAN + snap to the train pressure grid [COMPETITION-SPECIFIC?])
- PROCESS SLIP: I did not write the expected value down before running this comparison. The candidate sets were fixed before the run, and the primary candidate (C1) was named in the script before it ran.
- S7 reproduces sub07's logged fold-0 value, 0.19266, which confirms the row order.

| set | models (fold 0) | mean | mean+snap |
|---|---|---|---|
| S7 (sub07 fold-0 set) | exp04, 06, 07, 08 | 0.19409 | 0.19266 |
| S8 (sub08 fold-0 set) | S7 + exp09 | 0.19339 | 0.19193 |
| C1 primary | S8 + exp12a/b/c | 0.18429 | 0.18270 |
| C2 | exp12a/b/c only | 0.18517 | 0.18355 |
| C3 | C1 + exp13b/c + exp11a/b | 0.18399 | 0.18240 |

- sub09 = C1: runs/ensemble_sub09_stdout.txt, submissions/sub09_S8_plus_N1_mean_snap.csv (md5 512bc61d).
- The test average uses 12 files: exp04 f0, exp06 f0-4, exp07, exp08, exp09, exp12a/b/c.
- Format checks passed: shape (603600, 2); columns id, pressure; ids in sample_submission order; 0 NaN; every value on the grid. Mean abs diff vs sub07 on u_out==0 rows is 0.0287.
- NOT SUBMITTED; asking the user first.
- Caveat: N1 is 3 of 8 models on fold 0 but 3 of 12 test files, so the fold-0 gain (-0.010 vs S7) probably overstates the LB gain. C3 was 0.0003 better than C1 on fold 0; I did not pick it, because choosing by a 0.0003 fold-0 difference would overfit.

### Timing: bs 64 (for queue planning)
- Measured with a 3-epoch run, runs/bench_bs64.
- Alone: 20.5 s/epoch, so about 104 min per 300-epoch run.
- With src/eval_train.py (8 CPU threads) running at the same time: 31.0 s/epoch, +51% (runs/bench_bs64_contended).
- My guess was 18-20 s.
- CORRECTION to the earlier note: CPU jobs DO slow small-batch training. The earlier finding (no slowdown) was for bs 512. Rule from now on: no CPU jobs while a bs <= 128 run trains.
- For comparison, bs 128 (exp12a) runs at 16.1-16.8 s/epoch.


### Saturday day queue: PROPOSED, NOT STARTED (awaiting the user's approval)
- Script: src/satqueue.sh (md5 e65975e9). Server container started 19:36:23 UTC. Planned stop 01:36 UTC = 5:36 PM PT; the 6 h limit is UNCONFIRMED.
- New reference: N1 (exp12a/b/c, bs 128), mean 0.20490. Pooled s 0.00453 (N1 SD 0.00364, old controls 0.00527).
- Rule for a 2-seed arm: ADOPT if mean < 0.20490 - 2*0.00453*sqrt(1/2 + 1/3) = 0.19664 (detectable effect 0.0083).

| order | run | change vs reference | seeds | GPU |
|---|---|---|---|---|
| 1 | exp16a | D1: bs 128 -> 64, LR 2e-3 unchanged | 42 | ~104 min (20.5 s/epoch measured) |
| 2 | exp16b | same | 43 | ~104 min |
| 3 | exp12a fold 1 | none (the reference config, seed 42, on fold 1) | 42 | ~82 min |

Total about 290 min. A 20:00 UTC start ends about 00:50 UTC (4:50 PM PT), 46 min before the planned stop.

- D1 pre-registration:
  - Hypothesis: N1 helped through more optimizer steps (train MAE 0.098 -> 0.057, val better too), and halving the batch again (8x the steps of bs 512) continues the trend. Not competition-specific. It targets the per-breath level offset only indirectly: bs 128 cut the level part by 6% and the shape part by 9%.
  - Expected fold-0 OOF (GUESS): 2-seed mean 0.198 (range 0.190-0.212).
  - Proven wrong if: mean >= 0.20490, i.e. no gain beyond bs 128. Between 0.19664 and 0.20490 is inconclusive and not adopted. Instability (a val spike that does not recover) also counts against.
  - Secondary readout: mean |offset| and level share, compared with N1 (0.1527 / 0.367).
- exp12a fold 1 pre-registration:
  - Purpose: test predictions for the final ensemble, and a replication of N1 on a second fold.
  - Expected (GUESS): fold-1 OOF 0.200-0.208, vs exp06 fold 1 at 0.21514.
  - Proven wrong (the bs 128 gain does not carry over) if fold-1 OOF >= 0.2101, i.e. less than 0.005 better than exp06 fold 1. This is one run vs one run, so the seed-noise SE of the difference is about 0.007; read with care.
- Why there are NOT 2 more seeds of the "stacked config": only N1 passed, so the stacked config IS N1, which already has 3 seeds. The slot goes to fold 1 instead.
- Why no second level-targeted idea today: the only level-specific candidate with some support is more capacity (the offset is shared across models, which points at bias). That is hidden 256: 2 seeds x ~170 min at bs 128 (GUESS), which does not fit. Proposed for tonight's 12 h pod.


## Sat Nov 7 afternoon: sub09 submitted, Saturday queue started
- The user approved both ("Approved: submit sub09, and start the Saturday queue now").
- src/satqueue.sh (md5 e65975e9) started 20:19:25 UTC.
- Measured bs 64 speed in exp16a: 19.6 s/epoch, about 99 min per run.
- Projected ends: exp16a ~21:59, exp16b ~23:39, exp12a fold 1 ~01:02 UTC. That leaves 34 min before the planned (unconfirmed) 01:36 UTC stop.

| when (UTC) | what | OOF fold 0 | LB | evidence |
|---|---|---|---|---|
| 2026-11-07 20:19:44 | sub09 = sub08's models + exp12a/b/c (bs 128), MEAN+snap, submissions/sub09_S8_plus_N1_mean_snap.csv (md5 512bc61d) | 0.18270 | **0.17685 (new best; sub07 was 0.18125)** | CLI "Score: 0.17685"; git fa7205d. Approved by the user |

- The fold-0 gain vs sub07 was -0.0100; the LB gain is -0.0044, as the caveat expected (N1 is 3 of 12 test files).

## Note: the train/val gap GREW while validation improved (contradicts our starting framing)
- N1 (bs 128): eval-mode train MAE 0.057 vs val 0.205 (ratio about 3.5). The bs-512 controls: train 0.098 vs val 0.221 (ratio about 2.2).
- So the gap grew by about 0.025 in absolute terms, and validation still improved by 0.016.
- This contradicts the "close the train/val gap" framing we started with, which treated the gap as memorization to be shrunk. That framing drove N2 (weight decay), N4 (input noise), exp05 (dropout), and the call to DROP hidden 256. All three regularizers failed to beat the controls.
- A large gap is not the problem by itself. What predicts validation here is how well the model is optimized. Lower train error came WITH lower val error.
- Best-epoch check: N1's best val epochs were 266, 261 and (exp12c) late in the run. The last-10-epoch minimum is within 0.0002 of the best. So there is no late-epoch overfitting either.

## Reconciling the capacity reasoning (hidden 256)
- Yesterday's argument (section "T3 (hidden 256): recommendation DROP"): the failure is generalization of a memorized breath-level quantity, not underfitting (train 0.098), so more capacity mainly adds memorization room.
- What changed:
  - (1) N1 cut train error 0.098 -> 0.057 and val improved. The premise "lower train error means more memorization and worse val" is contradicted in this regime.
  - (2) The level offset is mostly SHARED across seeds (corr 0.76-0.80; averaging seeds removes shape error faster than level error). Shared error is bias of the model class, not seed variance. Capacity is a bias lever.
  - (3) A comparison I MISSED (the data existed once T2 finished on Nov 6):
    - exp02 = hidden 256, bs 512, lr 2e-3, **120** epochs: 0.24189.
    - exp10a/b = hidden 128, bs 512, lr 2e-3, **150** epochs: 0.25689 / 0.25429 (mean 0.2556).
    - So hidden 256 beat hidden 128 by about 0.014 with FEWER epochs.
    - Confounds: one seed for hidden 256; v1 vs v2 script (same 31 features, same split code; v1 had wd 1e-5 hard-coded = v2's default).
    - Yesterday's exp02-vs-exp04 comparison (0.242 vs 0.216) compared 120 with 300 epochs, and epochs turned out to matter a lot (T2: +0.035 for 150 vs 300).
- Conclusion: the case for hidden 256 is now positive, not negative. It goes first on tonight's pod.

### Saturday night pod queue: DRAFT, NOT LAUNCHED (needs the user's approval)
- Script: src/podqueue_sat.sh (md5 b71bea51). Decision helper: src/decide_vs_ref.py (md5 060d2bbe). The helper reproduced the hand-computed threshold 0.19664 on a dry run and returns INCOMPLETE while exp16a/b are unfinished.
- The batch size is picked automatically by the pre-registered D1 rule: bs 64 if D1 ADOPTS, otherwise bs 128. Today's measured result supports bs 128.

| order | run | change | seed | GPU if bs 128 | GPU if bs 64 |
|---|---|---|---|---|---|
| 1 | exp18a | H1: hidden 128 -> 256 | 42 | ~160 min (GUESS: 29.3 s/epoch measured at bs 512 x 1.07, the bs-128 slowdown seen at hidden 128) | ~190 (GUESS) |
| 2 | exp18b | same | 43 | ~160 (GUESS) | ~190 (GUESS) |
| 3 | exp12a fold 1 | none; only if today's fold-1 run did not finish | 42 | 82 | (exp16a fold 1) 99 |
| 4-6 | reference config folds 2, 3, 4 | none (final-ensemble coverage) | 42 | 3 x 82 | 3 x 99 |

- Total if bs 128: 566 min (648 if fold 1 has to be rerun), on a 720-min pod.
- Total if bs 64: about 776 min. That is over 12 h, so fold 4 would be cut off. Each run saves when it finishes, so only the run in progress would be lost.

- H1 pre-registration:
  - Hypothesis: the model is bias- and optimization-limited, not memorization-limited. More capacity lowers the shared (systematic) error, including the per-breath level offset. Not competition-specific.
  - Change: hidden 128 -> 256, at the reference batch size. Nothing else changes.
  - Expected fold-0 OOF (GUESS): 2-seed mean = reference - 0.008 (range -0.020 to +0.010).
  - Rule: ADOPT if mean < reference mean - 2*0.00453*sqrt(1/2 + 1/3). If bs 128 is the reference, that is 0.19664.
  - Proven wrong if the mean is >= the reference mean (no gain). Also proven wrong on the mechanism if mean |offset| falls by no more than the 6% that bs 128 gave (the hypothesis says capacity hits the shared/level part specifically).
  - Also report: the pair-average gain (2-seed mean minus their average). It estimates this config's seed-variance term d for the budget model below.

## Final-ensemble compute budget (config locks Sunday night; runs Mon morning - Wed night)
- GPU hours available: ASSUMPTION/GUESS. One GPU per user. Mon day server (6 h) + Mon night pod (12 h) + Tue day (6 h) + Tue night (12 h) + Wed day (6 h) = 42 h nominal. With relaunch gaps and margins, plan on 30-34 h; 42 h is the ceiling.
- Per-run times (300 epochs, one fold, including ~1 min data load):
  - bs 128 / h128: 82 min, MEASURED (exp12a/b/c).
  - bs 64 / h128: ~100 min, MEASURED (19.6 s/epoch in exp16a).
  - h256 / bs 128: ~160 min, GUESS.
  - h256 / bs 64: ~190 min, GUESS.

| candidate | runs in 30 h | 34 h | 42 h | full 5-fold sets at 34 h |
|---|---|---|---|---|
| bs128 / h128 | 21 | 24 | 30 | 4.8 |
| bs64 / h128 | 18 | 20 | 25 | 4.0 |
| h256 / bs128 (GUESS time) | 11 | 12 | 15 | 2.4 |
| h256 / bs64 (GUESS time) | 9 | 10 | 13 | 2.0 |

### Averaging curve on fold 0 (runs/analysis/averaging_curve.py, .txt; mean of all subsets of size n, no snap)
- bs-512 h128 family (8 models: exp06/08/09/07/13b/13c/11a/11b): n=1 0.2187, 2 0.2032, 3 0.1976, 4 0.1946, 5 0.1928, 6 0.1915, 7 0.1906, 8 0.1899.
  - MAE(n) = c + d/n fits within about 0.001 at every n: c 0.186, d 0.033.
- N1 bs 128 (3 models): n=1 0.2049, 2 0.1904, 3 0.1852. Fit: c 0.1753, d 0.0296. Extrapolating beyond n=3 is a GUESS.
- Mixing: N1 plus 1-5 weaker bs-512 models gives 0.1835-0.1851 vs N1 alone 0.1852. Weaker models add almost nothing once there are 3 strong ones.
- Estimate (model, GUESS beyond the data): E(K) = s1 - d(1 - 1/K), with d about 0.03 for both families measured.
  - bs128 at K = 24: 0.1753 + 0.0296/24 = about 0.1765 (fold-0 proxy, no snap).
  - The 1/K term left over is only 0.0012-0.0030 for K = 10-24. Config differences we have measured are much bigger (0.016 for bs).
  - Breakeven single-model fold-0 OOF to beat bs128 at 24 runs:
    - bs64 at 20 runs: < 0.2046, about 0.0003 better than N1.
    - h256/bs128 at 12 runs: < 0.2036 (0.0013 better).
    - h256/bs64 at 10 runs: < 0.2031 (0.0018 better).
- Answer (inference): we are in the FEWER-STRONGER regime. Pick the config with the best single-model error, even at half the run count.
- But a 2-seed arm's SE (~0.004) is larger than these breakevens. So: switch config only on a rule pass; otherwise keep bs128 (measured, cheapest). A mixed final ensemble costs almost nothing (see the mixing rows) and hedges.
- Caveats:
  - (a) d is assumed equal across configs. If h256 seeds are more correlated (smaller d), its plateau sits closer to its single-model error. The 2-seed pair average will show this.
  - (b) The test ensemble mixes models trained on different folds, which is probably more diverse than same-fold seeds. That shifts everyone's curve similarly.
  - (c) The fold-0 -> LB mapping is not stable: sub07 -0.0114, sub09 -0.0059.
- What would change the answer:
  - D1 passes (bs64 mean < 0.19664) -> bs64 becomes the base config.
  - D1 mean >= 0.2049 -> the steps story stops at bs128.
  - H1 passes -> h256 wins even at ~12 runs.
  - H1's pair-average gain far below 0.03 (say < 0.015) -> its fewer-models penalty and plateau are worse than modelled; prefer more h128 models unless H1's single-model gain is large.
  - exp12a fold 1 >= 0.2101 (the bs128 gain does not transfer to fold 1) -> every fold-0-based choice needs re-checking before the lock.
  - Sunday's GPU hours are NOT in this budget. If tonight settles the config, Sunday adds about 10-13 more runs.


## Night pod queue APPROVED (user, Sat Nov 7) + speed logging added
- The user approved src/podqueue_sat.sh as drafted, and asked that it log the measured seconds/epoch for hidden 256 after the first 10 epochs (160 min is a GUESS).
- Change made (no other edits): run() now starts training in the background and, once 11 epoch lines exist, logs "SPEED <run> :: median X s/epoch over epochs 1-10 -> projected Y min for 300 epochs". Epoch 0 is skipped because it includes warm-up. It logs this for every run; if a run ends before a check, it logs it at the end.
- Tests:
  - On real logs: exp12a 16.8 s (full-run median 16.1, so the first-10 read is about 4% high); exp16a 20.5; exp02 (hidden 256, bs 512) 29.2.
  - Fake trainer: correct median, logged mid-run, END line written.
  - bash -n passes. New md5 is in the commit.
- Launch timing: exp16a started 20:19:25 UTC at 19.8 s/epoch -> ends ~22:00. exp16b ~23:40. exp12a fold 1 (82 min) -> ~01:02 UTC = 5:02 PM PT.
- Launch window: from "satqueue done" (~5:05 PM PT) to the planned server stop (5:36 PM PT, unconfirmed). If the server dies early, launch anyway; the script reruns fold 1 if needed.
- The pod (12 h) would end ~5:05 AM PT Sunday. If bs 128 and the hidden-256 guess hold, the queue ends ~2:35 AM PT.

## Sunday plan (pre-registered Sat Nov 7, before any H1/D1 result is known)
### Sunday daytime (DataHub server, ~330 usable GPU-min)
- "Closest to winning" = the arm (H1 hidden 256, D1 bs 64) with the lowest diff = 2-seed mean - reference mean, provided diff < 0. If neither diff is < 0, nothing is close.
- Case A (D1 REJECT; H1 ran at bs 128):
  - The closest arm gets seed 44: exp18c (h256/bs128, ~160 min GUESS) or exp16c (bs64, ~100 min). This applies even if it already passed on 2 seeds, so the lock rests on 3.
  - The rest of the day goes to reference-config folds (exp12b folds 1, 2, ... at 82 min each).
  - If nothing is close: all slots go to reference folds.
- Case B (D1 ADOPT; the reference is bs 64 with 2 seeds; H1 ran at bs 64):
  - exp16c (D1 seed 44) first, so the bs-64 decision rests on 3 seeds.
  - Then exp18c (H1 seed 44 at bs 64, ~190 min GUESS) if H1's diff vs the bs-64 reference is < 0.
  - Otherwise reference folds.
### Sunday evening: LOCK by these rules
- Arm adopted if mean over ALL its seeds < ref_mean - 2*0.00453*sqrt(1/n + 1/n_ref). With 3 vs 3 seeds against N1, the threshold is 0.20490 - 0.00740 = 0.19750.
- A third seed CAN overturn a 2-seed pass.
- If H1 and D1 are both adopted in case A (tested separately), the untested combination is NOT stacked. Pick the one with the lower budget-model estimate E(K) = s1 - d(1 - 1/K), using its K from the table below and its own measured d (from its 2-3 seed subsets).
- In case B, adopting H1 means h256/bs64.
- Nothing adopted -> lock bs128/h128.
- I will write and commit the lock script BEFORE reading the Sunday third-seed result.
### Sunday night: first final-ensemble pod
- Launched Sunday night, not Monday: the locked config, seeds 43/44 over folds 0-4, in an order that fills missing folds first.

### Final-ensemble budget REWRITTEN: Sunday-night pod + Mon-Wed
- GPU hours (ASSUMPTION/GUESS): Sunday-night pod about 11 usable h (12 nominal), plus Mon-Wed 30-34 h. Planned 41-45 h; ceiling 54 h.

| config | min/run | runs at 41 h | 45 h | 54 h | 5-fold sets at 45 h | breakeven single-model fold-0 OOF vs bs128 at 45 h |
|---|---|---|---|---|---|---|
| bs128 / h128 | 82 (measured) | 30 | 32 | 39 | 6.4 | - (0.2049 measured) |
| bs64 / h128 | 100 (measured) | 24 | 27 | 32 | 5.4 | < 0.2047 |
| h256 / bs128 | 160 (GUESS; tonight's SPEED line replaces it) | 15 | 16 | 20 | 3.2 | < 0.2040 |
| h256 / bs64 | 190 (GUESS) | 12 | 14 | 17 | 2.8 | < 0.2037 |

- Model (GUESS beyond the data): bs128 at K = 32 -> 0.1753 + 0.0296/32 = 0.1762 on the fold-0 proxy, no snap.
- Already banked (not counted above), if bs 128 is locked:
  - Reference seed 42 on folds 0-4 (exp12a f0 + today's f1 + tonight's f2-f4).
  - exp12b/c on f0.
  - Sunday-daytime fill folds.
- The conclusion is unchanged: the 1/K term left over is 0.0009-0.0021, so any real single-model gain of about 0.002+ wins. Switch config only on a rule pass.


## Sun Nov 8: night pod results (podqueue_sat.sh, 01:21-12:18 UTC) and Saturday day-queue results
- Completion: every run printed "FOLD k DONE".
  - Day queue: exp16a f0, exp16b f0, exp12a f1 (satqueue done 01:02 UTC).
  - Pod: exp18a f0, exp18b f0, exp12a f2, f3, f4 (podqueue done 12:18 UTC).
  - D1 was REJECT, so the pod ran at bs 128. The fold-1 rerun was skipped because exp12a f1 had finished.

| arm | runs (fold-0 OOF) | mean | diff vs N1 0.20490 | threshold (2 vs 3 seeds) | pre-registered expectation | verdict |
|---|---|---|---|---|---|---|
| D1 bs 64 | exp16a 0.20241, exp16b 0.20615 | 0.20428 | -0.00062 | 0.19664 | 0.198 (range 0.190-0.212); proven wrong if >= 0.20490 | NOT ADOPTED, inconclusive (between threshold and reference). My 0.198 guess was too optimistic. |
| H1 hidden 256 (bs 128) | exp18a 0.21552, exp18b 0.21455 | 0.21504 | +0.01014 | 0.19664 | ref - 0.008 (range -0.020 to +0.010); proven wrong if >= ref | REJECT, PROVEN WRONG. Worse by 0.010, at the edge of my range. |

- Hidden-256 speed, MEASURED: SPEED lines 42.6 / 41.0 s/epoch (epochs 1-10). Whole runs took 206 and 203 min wall (01:21-04:47, 04:47-08:10). My 160-min guess was WRONG by about 28%.
- So yesterday's capacity reasoning (bias-limited -> more capacity helps) failed its test. At bs 128 and 300 epochs, hidden 256 is worse than hidden 128.
  - The exp02-vs-exp10 comparison that motivated it was a single seed with a script change, and it did not replicate.
  - Possible reason (GUESS, untested): hidden 256 needs a different LR or more epochs. Not pursued; no budget.
- Reference config on other folds: exp12a (bs 128, seed 42) f1 0.19601, f2 0.20141, f3 0.20241, f4 0.19681. exp06 (bs 512) on the same folds: 0.21514 / 0.21981 / 0.21442 / 0.21492. Gains: -0.019 / -0.018 / -0.012 / -0.018.
  - Fold-1 pre-registration ("proven wrong if >= 0.2101"): NOT wrong. The bs-128 gain transfers to every fold.
- PROCESS NOTE: I read the 2-seed D1 and H1 results before the lock script existed (the user asked for them). The lock script (src/decide_lock.py, md5 7a07a50d, commit 51b6040) was committed before exp16c existed. exp16c is the only result it decides on that I have not seen.

## Sunday day queue (pre-registered plan, started 18:55 UTC)
- "Closest to winning" = the lowest diff, provided diff < 0. D1 -0.00062 qualifies; H1 +0.01014 does not. So exp16c (D1 seed 44) runs, then reference folds exp12b f1 and f2 (seed 43). Script: src/sunqueue.sh (md5 1489b5f6).
- Server container started 18:53:27 UTC; planned stop 00:53 UTC = 4:53 PM PT (UNCONFIRMED).
- Projected ends: exp16c ~20:35, exp12b f1 ~22:00, exp12b f2 ~23:25 UTC (3:25 PM PT).
- Honest note on exp16c's decision value: for D1 to pass with 3 seeds (threshold 0.19750), exp16c would need <= 0.18394. No single run has come close to that (best 0.19601), so the lock is all but certain to be bs128/h128. The run follows the pre-registered rule anyway. Its OOF and test predictions still go into the final ensemble.

## Ensemble pre-registration (written before computing)
- C1 = sub09's run list (exp04, 06, 07, 08, 09, 12a, 12b, 12c). Its test average now also includes exp12a f1-f4, because ensemble.py uses every finished fold.
- C4 (primary) = C1 + exp16a, exp16b (bs 64, near-N1 quality).
- C5 = C4 + exp18a, exp18b (hidden 256: weaker, but a different architecture).
- Expected fold-0 MEAN+snap (GUESS): C4 0.1812, C5 0.1808. C1's fold-0 value stays 0.18270 (same fold-0 models as sub09).
- Rule:
  - Build/propose C4 if it beats 0.18270 by >= 0.001.
  - Prefer C5 over C4 only if C5 is >= 0.001 better than C4.
  - If neither beats 0.18270 by >= 0.001: still propose C1-rebuilt (sub09 + exp12a f1-f4 test predictions), because the folds 1-4 OOF rows will show whether adding exp12a helps there.
- CPU note: computed with OMP_NUM_THREADS=1 while exp16c (bs 64) trains. Expect a slowdown of a minute or two.


### Ensemble result (runs/analysis/sun_ensemble.py, .txt)

| set | fold-0 models | mean | mean+snap | pre-registered expectation |
|---|---|---|---|---|
| C1 (sub09's list) | 8 | 0.18429 | 0.18270 | unchanged (same fold-0 models) |
| C4 primary (+ exp16a/b) | 10 | 0.18122 | **0.17957** | 0.1812 (GUESS) -> better than expected |
| C5 (+ exp18a/b, h256) | 12 | 0.18129 | 0.17968 | 0.1808 (GUESS) -> wrong: h256 models do not help |
| N1+D1 only (diagnostic, NOT a pick candidate) | 5 | 0.18052 | 0.17894 | - |

- By the rule: C4 beats 0.18270 by 0.0031 (>= 0.001), so build C4. C5 is not >= 0.001 better than C4, so not chosen.
- The diagnostic "N1+D1 only" is 0.0006 better than C4. It was not pre-registered as a pick and is under 0.001, so it is not chosen. It does hint that the old bs-512 models add nothing once there are 5 strong ones (consistent with the mixing analysis).
- Folds 1-4, test-side evidence for the exp12a f1-f4 predictions now in the average (mean+snap):

| fold | exp06 alone | exp12a alone | exp06 + exp12a |
|---|---|---|---|
| 1 | 0.21392 | 0.19464 | 0.18819 |
| 2 | 0.21862 | 0.20011 | 0.19285 |
| 3 | 0.21315 | 0.20113 | 0.19113 |
| 4 | 0.21366 | 0.19544 | 0.18887 |

So adding exp12a gains 0.022-0.026 on every fold.
- sub10 = C4, built with ensemble.py (mean, --snap). The test average uses 18 files: exp04 f0, exp06 f0-4, exp07, exp08, exp09, exp12a f0-4, exp12b f0, exp12c f0, exp16a f0, exp16b f0.
  - Saved as submissions/sub10_C4_mean_snap.csv (md5 e4cf020c), and as submission.csv.
  - Format checks passed: (603600, 2); id, pressure; ids in sample order; 0 NaN; all values on the grid. Mean abs diff vs sub09 on u_out==0 rows is 0.0252.
  - NOT SUBMITTED; asking the user.

### Tonight's final-ensemble pod: DRAFT (needs the user's approval)
- Script: src/podqueue_final1.sh (md5 79f36be7). At start it runs src/decide_lock.py and refuses unless the lock is bs128/h128 (for any other lock a new plan is needed).
- Order fills missing folds first: exp12b (seed 43) f3, f4; exp12c (seed 44) f1-f4; exp12d (seed 45) f0, f1. It also reruns exp12b f1/f2 if today's runs did not finish.
- Time: 8 runs x ~82 min (pod wall-clock last night: 80 / 83 / 82 min) = ~656 min on a 720-min pod.
- After it finishes, complete 5-fold sets: seed 42 (exp12a), seed 43 (exp12b), seed 44 (exp12c), plus exp12d f0-f1.
- Launch window: once "sunqueue done" is printed (projected ~23:20 UTC = 3:20 PM PT) and before the planned server stop (00:53 UTC = 4:53 PM PT, unconfirmed). A 3:25 PM PT launch would finish the queue ~2:25 AM PT; the pod expires ~3:25 AM PT.
- Budget update: the hidden-256 row is moot (H1 rejected). bs128 runs at 82 min measured.


| when (UTC) | what | OOF fold 0 | LB | evidence |
|---|---|---|---|---|
| 2026-11-08 18:58:37 | sub10 = C4 (sub09 models + exp16a/b + exp12a f1-f4), MEAN+snap, submissions/sub10_C4_mean_snap.csv (md5 e4cf020c) | 0.17957 | **0.17044 (new best; sub09 was 0.17685)** | CLI "Score: 0.17044"; git aeef04c. Approved by the user ("Yes, submit sub10") |

- The fold-0 gain vs sub09 was -0.0031; the LB gain is -0.0064. The LB gained more than fold 0, which fits the test-side exp12a f1-f4 evidence (fold 0 cannot see those models).
- The pod script approval is still pending; the user has not answered that yet.

- Pod script src/podqueue_final1.sh APPROVED by the user ("Approve pod script"). The only edit is the header comment (DRAFT -> APPROVED); md5 in this commit. The user launches it on a 12 h pod after "sunqueue done" (window 3:25-4:50 PM PT).


## Sunday queue results + LOCK (logged 2026-11-09 00:40 UTC)

sunqueue.sh finished: "sunqueue done Sun Nov  8 23:17:47 UTC 2026". Every run finished.

| run | config | fold | best val_insp_mae |
|---|---|---|---|
| exp16c | bs64/h128 seed 44 | 0 | 0.21574 |
| exp12b | bs128/h128 seed 43 | 1 | 0.21192 |
| exp12b | bs128/h128 seed 43 | 2 | 0.20578 |

- D1 (bs64) 3-seed fold-0 mean: (0.20241 + 0.20615 + 0.21574)/3 = 0.20810, vs N1 ref 0.20490 (+0.0032). The pre-registered bar needed exp16c <= 0.18394; it got 0.21574. Not adopted.
- decide_lock.py (md5 7a07a50d, committed in 51b6040 before exp16c existed) prints: LOCK bs128/h128.
- exp12b fold speed: 16.7-16.8 s/epoch (about 84 min/run), matching the 82-min estimate.
- GPU idle from 23:17 UTC until the pod launch (the queue ended about 1.1 h before this check).


## Final pod #1 results (logged 2026-11-09 ~17:35 UTC)

podqueue_final1.sh: "podqueue_final1 start Mon Nov  9 01:07:59 UTC 2026 :: LOCK bs128/h128" ... "podqueue_final1 done Mon Nov  9 12:12:05 UTC 2026". All 8 runs finished; runs/PODQUEUE_FINAL1_DONE exists. The pod then sat idle until its 12 h timeout (~13:08 UTC).

| run (bs128/h128) | seed | fold | best val_insp_mae |
|---|---|---|---|
| exp12b | 43 | 3 | 0.20365 |
| exp12b | 43 | 4 | 0.20265 |
| exp12c | 44 | 1 | 0.20553 |
| exp12c | 44 | 2 | 0.21602 |
| exp12c | 44 | 3 | 0.20061 |
| exp12c | 44 | 4 | 0.20925 |
| exp12d | 45 | 0 | 0.20762 |
| exp12d | 45 | 1 | 0.20456 |

- exp12a, exp12b, exp12c now each have all 5 folds. exp12d has folds 0-1.
- exp12d fold 0 (0.20762) is inside the N1 reference spread (0.20142-0.20869).
- Run time: 82-84 min per run, 11.07 h for the 8 runs.
- CORRECTION: on Sunday I told the user that skipping exp12b f1/f2 cut the queue to about 8.7 h. Wrong: those folds were never among the 8 runs; the real length was 11.07 h, 55 min under the 12 h timeout.
- GPU idle from 12:12 UTC (queue done) until the next queue starts; this server started 17:26 UTC.

## Monday ensemble check - PRE-REGISTRATION (written before running it)

One change vs sub10 (C4): add exp12d and exp16c as members (C6). The ensemble also picks up all new fold test files automatically (exp12b f1-f4, exp12c f1-f4, exp12d f0-f1), so 29 test files instead of 18.
- Expected fold-0 mean+snap for C6: 0.1785 (GUESS; range 0.1775-0.1795). C4 was 0.17957.
- Proven wrong if C6 fold-0 mean+snap >= 0.17957; then sub11 uses C4 members only (still with the new fold test files).
- Fold-0 OOF cannot see the extra test-time fold models, so also: full 5-fold OOF (folds 0-4) of mean(exp12a, exp12b, exp12c), no snap. Expected 0.185 from the averaging curve c + d/3 = 0.1753 + 0.0099 (estimate, not a guess). exp12a alone 5-fold OOF expected about 0.200.
- Snap is [COMPETITION-SPECIFIC?] (logged before).
- Decision: build sub11 from whichever of C6/C4 is lower on fold-0 mean+snap, run the format checks, ask the user before submitting.
