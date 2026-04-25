# POMO-CTSP-d Experiment Handoff

Last updated: 2026-04-25

This file records the experiment decisions and operational changes needed for
another AI or researcher to continue the undergraduate-thesis experiment without
reconstructing the project context from scratch.

## Research Positioning

The proposed model should be described as priority-group-aware rather than
spatial-cluster-aware. The main signal is priority/group structure:

- group id / priority group
- same-group relation
- priority distance
- d-relaxed feasibility structure

The main synthetic train/test distribution does not require nodes in the same
priority group to be spatially clustered.

## Fixed Main Setting

Use this setting for the thesis main line:

- problem_size = 100
- num_groups = 8
- relaxation_d = 1
- random coordinates
- random priority-group assignment

Do not make variable-d or random num_groups the default main training setup.
The code may keep those capabilities as optional extensions.

## Final Model Set

Final thesis model set after the 2026-04-25 discussion:

- baseline_n100_g8_d1:
  `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/checkpoint-best.pt`
- legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1:
  existing structural ablation result, reused as auxiliary evidence.
- legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1:
  existing structural ablation result, reused as auxiliary evidence.
- new_full_learnable_bias_n100_g8_d1:
  `CSTPd_cluster/POMO/train_n100_learnable_bias.py`
- scheduled_bias_ablation_n100_g8_d1:
  old full-cluster result, now interpreted as the hand-crafted
  scheduled/fixed-bias ablation.
- wo_all_bias_n100_g8_d1:
  `CSTPd_cluster/POMO/train_n100_wo_all_bias.py`

The two structural ablation scripts have been updated to the new learnable-bias
configuration for possible retraining:

- `CSTPd_cluster/POMO/train_n100_wo_group_embedding.py`
- `CSTPd_cluster/POMO/train_n100_wo_fusion_gate.py`

Their existing result folders remain legacy scheduled-bias runs; only new runs
from these scripts will be learnable-bias structural ablations.

## Added / Adjusted Utilities

- `scripts/generate_synthetic_test_dataset.py`
  Generates a fixed same-distribution synthetic test set and saves it as `.pt`
  plus a JSON sidecar.

- `scripts/evaluate_ctspd.py`
  Unified checkpoint evaluator for fixed synthetic test sets and external
  `.ctspd` benchmark instances. It writes:
  - `test_instances.csv`
  - `test_summary.csv`
  - `test_summary.json`

- `scripts/summarize_results.py`
  Collects multiple `test_summary.json` files into one `summary.csv` for paper
  tables. Missing fields are written as NaN.

- `scripts/run_training_queue.py`
  Runs training scripts sequentially in the background-friendly queue style.
  It does not change checkpoint or result-folder logic in the original trainers.

## Final Bias Design

The new full model initializes from the old scheduled/fixed attention bias and
learns residual relation bias:

```python
cluster_bias_mode = 'scheduled_residual'
same_group_bias_init = 0.1
same_group_bias_final = 1.25
same_group_bias_warmup_epochs = 20
priority_distance_bias = 0.15
relation_bias_mode = 'learnable'
relation_bias_init = 0.0
use_decoder_priority_bias = False
```

With zero residuals this is exactly the old hand-crafted setup:

```python
cluster_bias_mode = 'scheduled'
priority_distance_bias = 0.15
relation_bias_mode = 'none'
use_decoder_priority_bias = False
```

The true `w/o all bias` ablation keeps group embedding and the fusion gate but
turns off all attention/logit bias terms:

```python
cluster_bias_mode = 'none'
priority_distance_bias = 0.0
relation_bias_mode = 'none'
use_decoder_priority_bias = False
```

The old `w/o_cluster_bias` run is no longer part of the final model set because
it removed only the scheduled same-group bias while keeping the fixed
`priority_distance_bias = 0.15`.

## Recommended Fixed Test Set

Generate the main same-distribution test set with:

```bash
python scripts/generate_synthetic_test_dataset.py \
  --problem-size 100 \
  --num-groups 8 \
  --relaxation-d 1 \
  --instance-num 1000 \
  --seed 20260423
```

Default output:

```text
data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt
```

All main models should be evaluated on exactly this file.

## Evaluation Examples

Synthetic main test:

```bash
python scripts/evaluate_ctspd.py \
  --model-type cluster \
  --model-variant learnable_bias \
  --checkpoint CSTPd_cluster/POMO/result/<run>/checkpoint-best.pt \
  --mode synthetic \
  --dataset-file data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt
```

Use `--model-variant scheduled_bias` for the old scheduled/fixed-bias ablation
and `--model-variant wo_all_bias` for the true no-bias ablation.

External benchmark test, kept separate from synthetic main testing:

```bash
python scripts/evaluate_ctspd.py \
  --model-type cluster \
  --model-variant learnable_bias \
  --checkpoint CSTPd_cluster/POMO/result/<run>/checkpoint-best.pt \
  --mode benchmark \
  --instance-glob "CTSPd(SOTA)/INSTANCES/Cluster_large/*100-C-*-1-*.ctspd" \
  --lkh-reference lkh_reference.csv
```

The benchmark mode is external/generalization evaluation, not same-distribution
testing.

## Current Known State

- CUDA is available on this machine: NVIDIA GeForce RTX 5090.
- The older baseline n100 tracked result under:
  `CSTPd_bsl/POMO/result/19日17点_bsl_n100_160ep_best15.703/`
  was removed from tracked artifacts on 2026-04-24. It was an older,
  non-preferred baseline result and should not be used as the thesis-main
  baseline now that the 2026-04-21 baseline result is available.
- The thesis-main baseline result is valid and committed under:
  `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/checkpoint-best.pt`
  and `checkpoint-latest.pt`.
  - epoch 160 / 160
  - best_epoch = 155
  - best_value = 15.784607734985352
  - total_training_time_sec = 17400.975403547287
- The old full-cluster result is retained as the scheduled/fixed-bias ablation
  under:
  `CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1/checkpoint-best.pt`
  and `checkpoint-latest.pt`.
  - epoch 160 / 160
  - best_epoch = 159
  - best_value = 15.73443634338379
  - total_training_time_sec = 17658.182819128036
- The legacy structural ablation, `w/o_group_embedding`, completed
  successfully under:
  `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1/checkpoint-best.pt`
  and `checkpoint-latest.pt`.
  - epoch 160 / 160
  - best_epoch = 154
  - best_value = 15.895662044067382
  - total_training_time_sec = 17853.272482395172
  - avg_epoch_time_sec = 111.58295301496983
- The legacy structural ablation, `w/o_fusion_gate`, completed
  successfully under:
  `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1/checkpoint-best.pt`
  and `checkpoint-latest.pt`.
  - epoch 160 / 160
  - best_epoch = 156
  - best_value = 15.786031008911133
  - total_training_time_sec = 17210.27424120903
  - avg_epoch_time_sec = 107.56421400755644
- The previous `w/o_cluster_bias` result
  `CSTPd_cluster/POMO/result/24日_17点01分_cluster_n100_d1_wo_cluster_bias/`
  was removed from the final tracked model set on 2026-04-25. It is not a true
  `w/o all bias` run because it kept `priority_distance_bias = 0.15`.
- The best training score from the removed `w/o_cluster_bias` run
  (`15.72086676513672`) motivated replacing hand-crafted scheduled/fixed bias
  with the new learnable priority-relation bias design.
- `.gitignore` keeps generic `result/` outputs ignored but whitelists the
  publishable thesis result artifacts for the final baseline, scheduled-bias
  ablation, legacy structural ablations, and future learnable-bias runs:
  `checkpoint-best.pt`, `checkpoint-latest.pt`, `training_metrics.csv`,
  `training_progress.json`, latest curve images, `img/*.jpg`, and `src/*.py`.
- The required-ablation training queue completed on 2026-04-24:
  `training_runs/20260424_121503_custom_queue/queue_state.json`.
  Queue PID was `4702`.
  The queue ran:
  - `train_n100_wo_fusion_gate.py`
  - `train_n100_wo_cluster_bias.py`
  The old `train_n100_wo_priority_distance_bias.py` script remains in the repo,
  but the final design now controls both bias terms together through
  `train_n100_wo_all_bias.py`.
- New full learnable-bias training completed successfully under:
  `CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias/checkpoint-best.pt`
  and `checkpoint-latest.pt`.
  - epoch 160 / 160
  - best_epoch = 160
  - best_value = 15.724634099121094
  - train_score = 15.724634099121094
  - total_training_time_sec = 19769.08372759819
  - avg_epoch_time_sec = 123.55677329748869
  It improved the old scheduled/fixed-bias full training best
  (`15.73443634338379`) by about `0.0098`.

## Operation Log

- 2026-04-23: Repository was pulled into `/root/autodl-tmp`.
- 2026-04-23: Project structure and existing results were reviewed.
- 2026-04-23: Main thesis setting fixed as n100/g8/d1.
- 2026-04-23: Added fixed synthetic test-set generation, unified evaluation,
  result summarization, and sequential training queue utilities.
- 2026-04-23: Split `wo_cluster_bias` and `wo_priority_distance_bias` ablations.
- 2026-04-23: Generated fixed main synthetic test set:
  `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt`.
- 2026-04-23: Smoke-tested synthetic and benchmark evaluation with the tracked
  baseline checkpoint, then removed smoke outputs.
- 2026-04-23: Started ablation training queue with PID `10214`.
  Queue state:
  `training_runs/20260423_152219_custom_queue/queue_state.json`.
  First active job at start:
  `CSTPd_cluster/POMO/train_n100_wo_group_embedding.py`.
- 2026-04-23: Checked intended thesis baseline/full-cluster uploaded
  `checkpoint-best.pt` files; both were incomplete/corrupt and not committed.
  Updated `.gitignore` to whitelist the intended checkpoint/metric files after
  complete replacements are uploaded.
- 2026-04-23: Stopped the first ablation queue because it inherited enhanced
  modules (`relation_bias_mode='learnable'`, `use_decoder_priority_bias=True`)
  and was not directly comparable with the existing full-cluster checkpoint.
  Deleted its partial output:
  `CSTPd_cluster/POMO/result/23日_15点22分_cluster_n100_d1_wo_group_embedding`
  and its queue logs.
- 2026-04-23: Reverted `CSTPd_cluster/POMO/train_n100.py` main defaults to the
  existing full-cluster architecture:
  `relation_bias_mode='none'`, `use_decoder_priority_bias=False`.
- 2026-04-23: Restarted ablation queue with PID `16906`.
  Queue state:
  `training_runs/20260423_163443_custom_queue/queue_state.json`.
  First epoch of `w/o_group_embedding` took 112.75 sec, matching the previous
  full-cluster training speed.
- 2026-04-23: Fixed `utils.copy_all_src()` to copy only whitelisted project
  `.py` files and exclude `result/`, `.git/`, `.autodl/`, caches, data, and
  training/test output directories. This prevents result `src/` snapshots from
  being polluted with Python environment or third-party module files. Removed
  the already-polluted `src/` snapshot from the incomplete manually uploaded
  full-cluster result folder; checkpoint files are unaffected.
- 2026-04-23: The thesis-main baseline and full-cluster result folders were
  re-uploaded completely, verified with `torch.load`, strict state-dict loading,
  and `training_progress.json`/`training_metrics.csv` checks. Their
  `checkpoint-best.pt`, `checkpoint-latest.pt`, metrics, progress JSON, source
  snapshots, and training curve images were committed.
- 2026-04-23: Reconstructed the clean `src/` snapshot for
  `CSTPd_cluster/POMO/result/21日_12点17分_cluster_n100_d1_resume_e116_to160`
  after its polluted uploaded `src/` folder was intentionally omitted.
- 2026-04-23: Paused the ablation queue scheduler while allowing the active
  `train_n100_wo_group_embedding.py` child process to finish. A watcher cleaned
  up the stopped queue parent after the current model completed, preventing
  `w/o_fusion_gate` from starting automatically.
- 2026-04-23: `cluster_n100_d1_wo_group_embedding` completed 160 epochs.
  Final recorded result:
  - result folder:
    `CSTPd_cluster/POMO/result/23日_16点34分_cluster_n100_d1_wo_group_embedding`
  - best_epoch = 154
  - best_value = 15.895662044067382
  - latest epoch = 160
  - latest train_score = 15.896407746276855
  - total_training_time_sec = 17853.272482395172
  - avg_epoch_time_sec = 111.58295301496983
- 2026-04-24: Verified the two required remaining ablation configs before
  restarting training:
  - `w/o_fusion_gate`: keeps `use_group_embedding=True`,
    `cluster_bias_mode='scheduled'`, `priority_distance_bias=0.15`,
    `relation_bias_mode='none'`, and `use_decoder_priority_bias=False`; only
    sets `use_group_fusion_gate=False`.
  - `w/o_cluster_bias`: keeps `use_group_embedding=True`,
    `use_group_fusion_gate=True`, `priority_distance_bias=0.15`,
    `relation_bias_mode='none'`, and `use_decoder_priority_bias=False`; only
    sets `cluster_bias_mode='none'`.
- 2026-04-24: A first non-detached queue launch at
  `training_runs/20260424_121408_custom_queue/queue_state.json` did not leave
  a live training process. It can be ignored if present.
- 2026-04-24: Started the required-ablation queue in a detached session with
  PID `4702`.
  Queue state:
  `training_runs/20260424_121503_custom_queue/queue_state.json`.
  First active job:
  `CSTPd_cluster/POMO/train_n100_wo_fusion_gate.py`.
  Expected second job after the first completes:
  `CSTPd_cluster/POMO/train_n100_wo_cluster_bias.py`.
  First result folder created:
  `CSTPd_cluster/POMO/result/24日_12点15分_cluster_n100_d1_wo_fusion_gate`.
- 2026-04-24: `cluster_n100_d1_wo_fusion_gate` completed 160 epochs.
  Final recorded result:
  - result folder:
    `CSTPd_cluster/POMO/result/24日_12点15分_cluster_n100_d1_wo_fusion_gate`
  - best_epoch = 156
  - best_value = 15.786031008911133
  - latest epoch = 160
  - latest train_score = 15.787005732116699
  - total_training_time_sec = 17210.27424120903
  - avg_epoch_time_sec = 107.56421400755644
  The queue then automatically started
  `CSTPd_cluster/POMO/train_n100_wo_cluster_bias.py`; its result folder is
  `CSTPd_cluster/POMO/result/24日_17点01分_cluster_n100_d1_wo_cluster_bias`.
- 2026-04-24: `cluster_n100_d1_wo_cluster_bias` completed 160 epochs.
  Final recorded result:
  - result folder:
    `CSTPd_cluster/POMO/result/24日_17点01分_cluster_n100_d1_wo_cluster_bias`
  - best_epoch = 160
  - best_value = 15.72086676513672
  - latest epoch = 160
  - latest train_score = 15.72086676513672
  - total_training_time_sec = 16985.185193777084
  - avg_epoch_time_sec = 106.15740746110677
  The required-ablation queue then finished successfully.
- 2026-04-24: Committed the two required n100 ablation result folders in
  `f363a94`:
  - `cluster_n100_d1_wo_fusion_gate`
  - `cluster_n100_d1_wo_cluster_bias`
  The commit includes best/latest checkpoints, metrics, progress JSON, curve
  images, and source snapshots, while leaving numbered intermediate
  checkpoints, logs, queue state, and cache files untracked/ignored.
- 2026-04-24: Marked the old tracked baseline result
  `CSTPd_bsl/POMO/result/19日17点_bsl_n100_160ep_best15.703/` for deletion
  from the repository to reduce stale non-main artifacts. The thesis-main
  baseline remains
  `CSTPd_bsl/POMO/result/21日_13点43分_baseline_n100_d1/`.
- 2026-04-24: Noted the unexpected `w/o_cluster_bias` training-score result:
  it is better than the current full-cluster training best on this run. This
  may mean the scheduled same-group attention bias is over-constraining or
  simply that this seed favored the ablation. Do not rewrite the main model
  claim until fixed-test evaluation and, ideally, repeated-seed checks confirm
  the pattern.
- 2026-04-25: Final experiment set was revised to six models:
  baseline, two legacy structural ablations, new learnable-bias full model,
  old scheduled/fixed-bias ablation, and true `w/o all bias`.
- 2026-04-25: Added inverse-softplus initialization for learnable same-group
  bias so `same_group_bias_init` is the effective initial bias value.
- 2026-04-25: Added `train_n100_learnable_bias.py` for the new full model:
  `cluster_bias_mode='learnable'`, `priority_distance_bias=0.0`,
  `relation_bias_mode='learnable'`, and `use_decoder_priority_bias=False`.
- 2026-04-25: Updated `train_n100_wo_group_embedding.py` and
  `train_n100_wo_fusion_gate.py` to use the same learnable-bias setup for
  possible future retraining. Existing result folders remain legacy
  scheduled-bias structural ablations.
- 2026-04-25: Added `train_n100_wo_all_bias.py` for the true no-bias ablation.
- 2026-04-25: Renamed tracked n100 result folders:
  - baseline -> `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1`
  - old full -> `CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1`
  - old `w/o_group_embedding` ->
    `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1`
  - old `w/o_fusion_gate` ->
    `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1`
- 2026-04-25: Removed the old tracked `w/o_cluster_bias` result folder from the
  final model set because it is not a true `w/o all bias` ablation.
- 2026-04-25: Started new full learnable-bias training in a detached process:
  PID `6897`, log `training_runs/20260425_new_full_learnable_bias.log`,
  result folder
  `CSTPd_cluster/POMO/result/25日_11点14分_cluster_n100_d1_new_full_learnable_bias`.
- 2026-04-25: Stopped the first absolute-positive learnable-bias run at around
  epoch 102 because its best training score remained behind both the old
  scheduled full model and the old no-same-group-bias run. Reworked
  `train_n100_learnable_bias.py` in-place to use `cluster_bias_mode =
  'signed_learnable'`, `same_group_bias_init = 0.0`, `priority_distance_bias =
  0.15`, `relation_bias_init = 0.0`, and a 30-to-80 epoch residual-bias warmup.
  This makes the run initialize as the stronger no-same-group-bias model, then
  learn signed residual structure instead of forcing positive same-group bias.
- 2026-04-25: Restarted the reworked same full-model script in a detached
  process: PID `21877`, log
  `training_runs/20260425_new_full_learnable_bias_reworked.log`, result folder
  `CSTPd_cluster/POMO/result/25日_14点47分_cluster_n100_d1_new_full_learnable_bias_run02`.
- 2026-04-25: Stopped PID `21877` after clarifying that the intended new full
  model should not use the no-same-group-bias run as the base. Reworked
  `train_n100_learnable_bias.py` again to use `cluster_bias_mode =
  'scheduled_residual'`, keeping the old full model's same-group schedule
  (`0.1 -> 1.25` over 20 epochs) and fixed `priority_distance_bias = 0.15`,
  while learning zero-initialized residual same-group/relation bias. Verified
  that with zero residuals, attention bias matches the old full model exactly
  at epochs 1, 10, 20, 21, 33, 80, and 160.
- 2026-04-25: Restarted this old-bias-initialized learnable run in a detached
  process: PID `26302`, log
  `training_runs/20260425_new_full_learnable_bias_old_bias_init.log`, result
  folder
  `CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias`.
- 2026-04-25: The old-bias-initialized learnable run completed 160 epochs.
  Final recorded result:
  - best_epoch = 160
  - best_value = 15.724634099121094
  - latest epoch = 160
  - latest train_score = 15.724634099121094
  - total_training_time_sec = 19769.08372759819
  - avg_epoch_time_sec = 123.55677329748869
  Also updated `scripts/evaluate_ctspd.py` so checkpoints with
  zero-initialized residual same-group parameters and a scheduled runtime bias
  are loaded as `scheduled_residual` instead of plain positive `learnable`
  bias.

Update this section whenever long-running training or evaluation jobs are
started, stopped, or completed.
