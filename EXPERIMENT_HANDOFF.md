# POMO-CTSP-d Experiment Handoff

Last updated: 2026-04-23

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

## Main Models

Required:

- baseline_n100_d1: `CSTPd_bsl/POMO/train_n100.py`
- cluster_n100_d1: `CSTPd_cluster/POMO/train_n100.py`
- cluster_n100_d1_wo_group_embedding: `CSTPd_cluster/POMO/train_n100_wo_group_embedding.py`
- cluster_n100_d1_wo_fusion_gate: `CSTPd_cluster/POMO/train_n100_wo_fusion_gate.py`
- cluster_n100_d1_wo_cluster_bias: `CSTPd_cluster/POMO/train_n100_wo_cluster_bias.py`

Optional but useful:

- cluster_n100_d1_wo_priority_distance_bias:
  `CSTPd_cluster/POMO/train_n100_wo_priority_distance_bias.py`

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

## Important Correction

`CSTPd_cluster/POMO/train_n100_wo_cluster_bias.py` previously disabled both
`cluster_bias_mode` and `priority_distance_bias`. That mixed two ablations.
It now only sets:

```python
cluster_bias_mode = 'none'
priority_distance_bias = 0.15
```

The separate optional ablation
`train_n100_wo_priority_distance_bias.py` sets `priority_distance_bias = 0.0`.

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
  --model-variant full \
  --checkpoint CSTPd_cluster/POMO/result/<run>/checkpoint-best.pt \
  --mode synthetic \
  --dataset-file data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt
```

External benchmark test, kept separate from synthetic main testing:

```bash
python scripts/evaluate_ctspd.py \
  --model-type cluster \
  --model-variant full \
  --checkpoint CSTPd_cluster/POMO/result/<run>/checkpoint-best.pt \
  --mode benchmark \
  --instance-glob "CTSPd(SOTA)/INSTANCES/Cluster_large/*100-C-*-1-*.ctspd" \
  --lkh-reference lkh_reference.csv
```

The benchmark mode is external/generalization evaluation, not same-distribution
testing.

## Current Known State

- CUDA is available on this machine: NVIDIA GeForce RTX 5090.
- A tracked baseline n100 checkpoint currently exists under:
  `CSTPd_bsl/POMO/result/19日17点_bsl_n100_160ep_best15.703/checkpoint-best.pt`
- The intended thesis baseline checkpoint location is:
  `CSTPd_bsl/POMO/result/21日_13点43分_baseline_n100_d1/checkpoint-best.pt`
- The intended full cluster n100 checkpoint location is:
  `CSTPd_cluster/POMO/result/21日_12点17分_cluster_n100_d1_resume_e116_to160/checkpoint-best.pt`
- On 2026-04-23, the manually uploaded intended baseline and full-cluster
  `checkpoint-best.pt` files failed `torch.load` with:
  `PytorchStreamReader failed reading zip archive: failed finding central directory`.
  Treat those two uploaded checkpoint files as incomplete until they are
  replaced by complete uploads.
- `.gitignore` now keeps generic `result/` outputs ignored but whitelists the
  intended thesis baseline and full-cluster `checkpoint-best.pt`,
  `checkpoint-latest.pt`, `training_metrics.csv`, and `training_progress.json`
  files so complete replacements can be added normally.
- The current long-running training queue is for ablations only:
  - `train_n100_wo_group_embedding.py`
  - `train_n100_wo_fusion_gate.py`
  - `train_n100_wo_cluster_bias.py`
  - `train_n100_wo_priority_distance_bias.py`

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

Update this section whenever long-running training or evaluation jobs are
started, stopped, or completed.
