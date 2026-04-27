# POMO-CTSP-d Experiment Handoff

Last updated: 2026-04-27

This file records the experiment decisions and operational changes needed for
another AI or researcher to continue the undergraduate-thesis experiment without
reconstructing the project context from scratch.

## Recommended Reading for Continuation

Give a future AI or collaborator these files first:

1. `EXPERIMENT_HANDOFF.md`
   The authoritative current-state document: final model set, experiment
   framing, result paths, caveats, and operation log.
2. `README.md`
   Repository-level overview and basic project orientation.
3. `PROJECT_STRUCTURE.md`
   Cleaned repository layout and final reproduction flow.
4. `MODEL_ARTIFACTS.md`
   Final checkpoint/config/log/test-result inventory.
5. `RESULT_ARTIFACTS.md`
   Final dataset, result, LKH, and paper-artifact inventory.
6. `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts/README.md`
   Main same-distribution result tables/figures with the LOW_FIRST LKH
   benchmark and inference-time comparison.
7. `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts/FIGURE_HANDOFF.md`
   Final figure placement, caption, and downgrade/delete decisions.
8. `PROJECT_FILE_AUDIT.md`
   Cleanup audit and archive decisions.

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

Classical benchmark added on 2026-04-27:

- LOW_FIRST-patched LKH 3.0.14:
  `LKH-3.0.14/`, built with `make CTSPD_PRIORITY=LOW_FIRST`.
  This is not a trained neural model. It is used as a strong heuristic
  benchmark on the same fixed synthetic CTSP-d test set. The patch makes LKH
  follow this repository's convention that lower priority-group ids mean higher
  priority. The default upstream CTSP-D interpretation should be recovered by
  rebuilding without `CTSPD_PRIORITY=LOW_FIRST`.
  Note: LKH has upstream licensing terms in `LKH-3.0.14/README.txt`; be careful
  before redistributing the LKH folder outside the research context.

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

- `scripts/run_lkh_ctspd_benchmark.py`
  Exports the fixed synthetic `.pt` dataset to LKH-compatible `TYPE : CTSP-D`
  instances with explicit distance matrices, runs the LOW_FIRST-patched LKH
  executable, rotates/reverses returned tours to match the repository's
  feasibility checker, and writes:
  - `lkh_instances.csv`
  - `lkh_summary.json`
  - raw LKH tours, normalized tours, generated `.ctspd` instances, `.par`
    files, and LKH logs.

- `scripts/make_main_lkh_artifacts.py`
  Merges the six neural-model evaluations with the LKH benchmark into one
  `summary.csv` and paper-ready tables/figures, including cost, gap,
  pairwise-win, training-curve, and inference-time plots.

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

Concrete differences between the old and new full models:

- Same-group bias:
  the old full model uses only the fixed schedule `0.1 -> 1.25` over the first
  20 epochs. The new full model keeps this schedule as the base and adds a
  learnable scalar residual:
  `effective_same_group_bias = scheduled_bias + same_group_bias_param`.
  `same_group_bias_param` starts at `0.0`, is updated by the optimizer through
  normal backpropagation, and is clamped together with the scheduled value by
  `same_group_bias_max`.
- Priority-distance bias:
  both models keep the same fixed `priority_distance_bias = 0.15`; this part is
  intentionally unchanged.
- Relation attention bias:
  the old full model has no learned relation-bias table. The new full model
  adds a learnable per-head/per-group-distance relation attention bias, with
  `relation_bias_init = 0.0`, so it also starts from the old full model's
  behavior.
- Decoder priority bias:
  both models keep `use_decoder_priority_bias = False`.

Therefore the new full model is initialized to the old full model's attention
bias behavior, then learns only residual corrections on top of that prior. This
does not mathematically guarantee the final trained checkpoint must beat the old
one, because optimization can still move parameters poorly, but it does make the
initial model behavior equivalent and gives the model strictly more bias
capacity.

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

## Final Result Outputs

The current thesis-main outputs with the LKH benchmark produced on 2026-04-27 are:

- Main same-distribution synthetic evaluation:
  `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427`
  - inference setting:
    8-fold geometric augmentation, greedy/POMO decoding, best candidate
    selection over augmented directions and POMO starts, no stochastic sampling.
  - fixed dataset:
    `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt`
  - LKH setting:
    LOW_FIRST-patched LKH 3.0.14, `MAX_TRIALS=200`, `RUNS=1`,
    `SCALE=1000000`, `MAX_CANDIDATES=6`; all 1000 LKH tours were feasible.
  - combined summary:
    `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/summary.csv`
  - paper-ready outputs:
    `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts`
  - key files:
    `main_results_table.csv`, `main_results_table.md`, `ablation_table.csv`,
    `pairwise_win_counts.csv`, `pairwise_win_percent.csv`,
    `per_instance_costs.csv`, `average_cost_with_lkh_bar.png/.pdf`,
    `gap_to_lkh_bar.png/.pdf`, `per_instance_gap_to_lkh_boxplot.png/.pdf`,
    `pairwise_win_heatmap.png/.pdf`, `time_per_instance_bar_log.png/.pdf`,
    and `training_score_curves.png/.pdf`.

The previous main synthetic result directory
`test_results/thesis_main_synthetic_n100_g8_d1_20260426` was intentionally
deleted on 2026-04-27 to avoid confusion after adding LKH.

The older 2026-04-26 outputs were moved to local `archive_to_review/old_results/`
during the 2026-04-27 project cleanup. They remain useful context, but they are
not part of the final submitted reproducible artifact set:

- External LKH benchmark comparison:
  `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_20260426`
  - inference setting:
    `anchor,mds` reconstructed feature modes, 8-fold geometric augmentation on
    those features, greedy/POMO decoding, best candidate selection by the
    original CTSP-d distance matrix, no stochastic sampling or local search.
  - instances:
    archived `CTSPd(SOTA)/INSTANCES/Cluster_large/*100-C-*-1-*.ctspd`
  - LKH references:
    `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_20260426/lkh_reference.csv`
  - combined summary:
    `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_20260426/summary.csv`
  - paper-ready outputs:
    `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_20260426/paper_artifacts`
  - key files:
    `benchmark_results_table.csv`, `benchmark_results_table.md`,
    `per_instance_costs.csv`, `per_instance_winner_counts.csv`,
    `pairwise_win_table.csv`, `gap_to_lkh_bar.png/.pdf`,
    `per_instance_gap_heatmap.png/.pdf`, and
    `model_costs_vs_lkh.png/.pdf`.
- Enhanced external LKH benchmark stress test:
  `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_aug8_sample64_ls20_20260426`
  - inference setting:
    `anchor,mds` reconstructed feature modes, 8-fold geometric augmentation,
    greedy/POMO decoding plus 64 stochastic sampling runs per feature mode,
    best candidate selection by the original CTSP-d distance matrix, and
    same-priority swap local search with up to 20 passes.
  - combined summary:
    `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_aug8_sample64_ls20_20260426/summary.csv`
  - per-model outputs:
    `archive_to_review/old_results/test_results/thesis_benchmark_cluster_large_n100_d1_aug8_sample64_ls20_20260426/evaluations/<model>/`
  - each per-model output includes `test_instances.csv`, `test_summary.json`,
    and 10 saved final `.tour` files.
- Full learnable model-only inference exploration:
  `archive_to_review/old_results/test_results/full_learnable_model_only_best_of_ensemble_20260426`
  - inference setting:
    no model-weight changes and no traditional optimization/post-processing;
    candidates are generated only by the full learnable-bias model with
    feature-mode and softmax-temperature test-time sampling, then the best
    generated candidate per instance is selected by the original CTSP-d
    distance matrix.
  - supporting completed runs:
    `archive_to_review/old_results/test_results/full_learnable_model_only_parallel_ensemble_20260426`,
    `archive_to_review/old_results/test_results/full_learnable_model_only_mds_deep_sampling_20260426`, and
    `archive_to_review/old_results/test_results/full_learnable_model_only_temperature_sweep_20260426`.
  - key files:
    `README.md`, `summary.csv`, `summary.json`, and `best_instances.csv`.
- Reproducibility check:
  `archive_to_review/old_results/test_results/reproducibility_check_20260426`
  - `README.md`: human-readable pass/fail report.
  - `repro_check.json`: machine-readable checks and SHA256 checksums.
  - The check strictly loaded all six `checkpoint-best.pt` files on CPU,
    verified the fixed synthetic dataset metadata, verified all stored
    synthetic and benchmark result row counts/seeds/feasibility rates, validated
    paper figure file types, and dynamically reran the six-model benchmark
    evaluation to confirm exact stored `model_cost` reproduction.

Main same-distribution synthetic test averages on the 1000 fixed n100/g8/d1
instances after adding LKH:

| Model | Average cost | Gap to LKH | Time / instance | Speedup vs LKH | Feasible rate |
|---|---:|---:|---:|---:|---:|
| LOW_FIRST LKH | 15.132940 | 0.000% | 7.189s | 1.0x | 1.0 |
| Baseline | 15.494203 | 2.407% | 0.063s | 114.7x | 1.0 |
| New full learnable bias | 15.445869 | 2.087% | 0.069s | 104.7x | 1.0 |
| Scheduled/fixed bias | 15.452672 | 2.132% | 0.065s | 110.9x | 1.0 |
| True `w/o all bias` | 15.470992 | 2.254% | 0.063s | 113.4x | 1.0 |
| `w/o fusion gate` | 15.466143 | 2.222% | 0.064s | 111.8x | 1.0 |
| `w/o group embedding` | 15.699298 | 3.762% | 0.065s | 110.3x | 1.0 |

Interpretation of this table:

- LKH is a much stronger heuristic reference on the fixed same-distribution
  synthetic test set, but it is about two orders of magnitude slower than the
  learned policies under the recorded budget.
- Among neural models, the new full learnable-bias model remains the best
  average-cost model on the thesis-main synthetic distribution.
- The scheduled/fixed-bias ablation is very close to the full model, which
  should be discussed as an important nuance rather than hidden.
- The `w/o all bias` and structural ablations support the claim that explicit
  priority/group information is useful, with `w/o group embedding` clearly
  worse on the main synthetic distribution.

External LKH benchmark averages on the 10 `Cluster_large` n100/d1 instances:

| Model | Average cost | Average gap to LKH | Feasible rate |
|---|---:|---:|---:|
| Baseline | 29491.1 | 30.023362228761982% | 1.0 |
| New full learnable bias | 27134.5 | 19.64153639788954% | 1.0 |
| Scheduled/fixed bias | 26742.2 | 17.806783359242594% | 1.0 |
| True `w/o all bias` | 27291.9 | 20.34274308077898% | 1.0 |
| `w/o fusion gate` | 27978.9 | 23.349276883571527% | 1.0 |
| `w/o group embedding` | 26507.0 | 16.655800247413705% | 1.0 |

Enhanced external LKH benchmark averages with sampling64 plus same-priority
local search on the same 10 instances:

| Model | Average cost | Average gap to LKH | Feasible rate |
|---|---:|---:|---:|
| Baseline | 27038.5 | 19.107016748752358% | 1.0 |
| New full learnable bias | 25761.3 | 13.571954288993018% | 1.0 |
| Scheduled/fixed bias | 25305.0 | 11.509764257512131% | 1.0 |
| True `w/o all bias` | 25787.2 | 13.734386120421016% | 1.0 |
| `w/o fusion gate` | 26000.9 | 14.592376703089329% | 1.0 |
| `w/o group embedding` | 24272.4 | 6.88054973950998% | 1.0 |

Full learnable model-only test-time inference exploration on the same external
instances:

| Protocol | Average cost | Average gap to LKH | Feasible rate |
|---|---:|---:|---:|
| Full learnable, enhanced sampling64 + same-priority LS20 | 25761.3 | 13.571954288993018% | 1.0 |
| Full learnable, standard `anchor,mds` aug8 greedy | 27134.5 | 19.64153639788954% | 1.0 |
| Full learnable, model-only ensemble | 26574.1 | 17.206078282859504% | 1.0 |
| Scheduled/fixed, standard `anchor,mds` aug8 greedy | 26742.2 | 17.806783359242594% | 1.0 |
| `w/o group embedding`, standard `anchor,mds` aug8 greedy | 26507.0 | 16.655800247413705% | 1.0 |

The model-only ensemble changes only the test-time inference protocol for the
full learnable model. It does not alter model weights and does not use LKH,
local search, 2-opt, or same-priority swap search. It is useful as a deployment
strengthening result: full learnable becomes better than the scheduled/fixed
model under the standard external benchmark protocol, but it still does not
beat `w/o group embedding` under that standard protocol.

Important protocol distinction: the strongest recorded external benchmark
number for the full learnable model is still `25761.3` with average LKH gap
`13.571954288993018%`, but that number uses the enhanced sampling64 plus
same-priority local-search protocol. The model-only ensemble number
`26574.1` / `17.206078282859504%` should not be treated as replacing that
stronger enhanced result; it answers the narrower question of how far the full
model can be pushed when final tours must come directly from model-generated
candidates and no local-search post-processing is allowed.

Interpretation caution: the external benchmark is a transparent LKH comparison
and generalization stress test. These models were trained on the synthetic
n100/g8/d1 thesis distribution, not on TSPLIB-derived benchmark instances. The
1000-instance fixed synthetic result should remain the main ablation evidence.
The small external benchmark still favors the legacy `w/o group embedding`
model under both the greedy augmented and the sampling64+local-search settings,
which suggests distribution shift rather than overturning the same-distribution
ablation conclusion.

## Evaluation Examples

Synthetic main test:

```bash
python scripts/evaluate_ctspd.py \
  --model-type cluster \
  --model-variant learnable_bias \
  --checkpoint CSTPd_cluster/POMO/result/<run>/checkpoint-best.pt \
  --mode synthetic \
  --dataset-file data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt \
  --augmentation-factor 8
```

Use `--model-variant scheduled_bias` for the old scheduled/fixed-bias ablation
and `--model-variant wo_all_bias` for the true no-bias ablation.

External benchmark test, kept separate from synthetic main testing. This now
requires restoring the archived external data from
`archive_to_review/external_benchmarks/CTSPd(SOTA)/` or providing an equivalent
local path:

```bash
python scripts/evaluate_ctspd.py \
  --model-type cluster \
  --model-variant learnable_bias \
  --checkpoint CSTPd_cluster/POMO/result/<run>/checkpoint-best.pt \
  --mode benchmark \
  --instance-glob "archive_to_review/external_benchmarks/CTSPd(SOTA)/INSTANCES/Cluster_large/*100-C-*-1-*.ctspd" \
  --lkh-reference lkh_reference.csv \
  --augmentation-factor 8 \
  --benchmark-feature-modes anchor,mds
```

The benchmark mode is external/generalization evaluation, not same-distribution
testing.

LOW_FIRST LKH synthetic benchmark:

```bash
cd LKH-3.0.14
make CTSPD_PRIORITY=LOW_FIRST
cd ..
python scripts/run_lkh_ctspd_benchmark.py \
  --max-trials 200 \
  --runs 1 \
  --output-dir test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/lkh_low_first \
  --timeout-sec 120
```

Regenerate paper artifacts after all evaluations are present:

```bash
python scripts/make_main_lkh_artifacts.py \
  --result-dir test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427
```

## Current Known State

- CUDA is available on this machine: NVIDIA GeForce RTX 5090.
- The older baseline n100 tracked result under:
  `CSTPd_bsl/POMO/result/19日17点_bsl_n100_160ep_best15.703/`
  was removed from tracked artifacts on 2026-04-24. It was an older,
  non-preferred baseline result and should not be used as the thesis-main
  baseline now that the 2026-04-21 baseline result is available.
- The thesis-main baseline result is valid and committed under:
  `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/checkpoint-best.pt`.
  - epoch 160 / 160
  - best_epoch = 155
  - best_value = 15.784607734985352
  - total_training_time_sec = 17400.975403547287
- The old full-cluster result is retained as the scheduled/fixed-bias ablation
  under:
  `CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1/checkpoint-best.pt`.
  - epoch 160 / 160
  - best_epoch = 159
  - best_value = 15.73443634338379
  - total_training_time_sec = 17658.182819128036
- The legacy structural ablation, `w/o_group_embedding`, completed
  successfully under:
  `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1/checkpoint-best.pt`.
  - epoch 160 / 160
  - best_epoch = 154
  - best_value = 15.895662044067382
  - total_training_time_sec = 17853.272482395172
  - avg_epoch_time_sec = 111.58295301496983
- The legacy structural ablation, `w/o_fusion_gate`, completed
  successfully under:
  `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1/checkpoint-best.pt`.
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
  `checkpoint-best.pt`, `training_metrics.csv`, `training_progress.json`,
  latest score/loss curve images, and `src/*.py` snapshots.
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
  `CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias/checkpoint-best.pt`.
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
  snapshots, and training curve images were initially committed; the 2026-04-27
  cleanup later retained only `checkpoint-best.pt`, metrics/progress JSON,
  source snapshots, and latest score/loss preview images in the final tracked
  artifact set.
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
- 2026-04-26: Removed historical tracked `result/` artifacts that are not part
  of the final thesis model set:
  - early n20 baseline/cluster train and test result folders
  - early n20 generated tour folders under `CSTPd_bsl/POMO/result/tours` and
    `CSTPd_cluster/POMO/result/tours`
  This cleanup keeps the final n100/g8/d1 thesis result folders visible without
  older tracked result folders causing confusion.
- 2026-04-26: Started true `w/o all bias` training in a detached process:
  PID `4537`, log `training_runs/20260426_wo_all_bias.log`, result folder
  `CSTPd_cluster/POMO/result/26日_10点20分_cluster_n100_d1_wo_all_bias`.
  This run uses `cluster_bias_mode='none'`, `priority_distance_bias=0.0`,
  `relation_bias_mode='none'`, and `use_decoder_priority_bias=False`.
- 2026-04-26: True `w/o all bias` training completed successfully.
  Final recorded result:
  - result folder:
    `CSTPd_cluster/POMO/result/26日_10点20分_cluster_n100_d1_wo_all_bias`
  - epoch = 160 / 160
  - best_epoch = 157
  - best_value = 15.770512321472168
  - latest train_score = 15.773918809204101
  - total_training_time_sec = 17786.283062696457
  - avg_epoch_time_sec = 111.16426914185286
  The result is worse than the new full learnable-bias run
  (`15.724634099121094`) and the scheduled/fixed-bias ablation
  (`15.73443634338379`), so it supports keeping priority-relation bias in the
  final model design.
- 2026-04-26: Completed augmented fixed synthetic test evaluation for the final six-model
  set on:
  `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt`.
  Inference used 8-fold geometric augmentation and greedy/POMO best selection.
  Results are saved under:
  `test_results/thesis_main_synthetic_n100_g8_d1_20260426`.
  Paper-ready tables and figures are under:
  `test_results/thesis_main_synthetic_n100_g8_d1_20260426/paper_artifacts`.
  Main test averages:
  - baseline: `15.49420291519165`
  - new full learnable bias: `15.445868775367737`
  - scheduled/fixed bias: `15.452672064781188`
  - true `w/o all bias`: `15.470991799354554`
  - `w/o fusion gate`: `15.46614307975769`
  - `w/o group embedding`: `15.69929798603058`
  All six evaluations had `feasible_rate = 1.0`.
- 2026-04-26: Completed external benchmark instance testing for the final
  six-model set on the 10 `Cluster_large` n100/d1 `.ctspd` instances under:
  `CTSPd(SOTA)/INSTANCES/Cluster_large`.
  Inference used `anchor,mds` reconstructed feature modes, 8-fold geometric
  augmentation on those features, and greedy/POMO best selection by the original
  CTSP-d distance matrix.
  LKH reference costs were parsed from:
  `CTSPd(SOTA)/TOURS/Cluster_large`.
  Results are saved under:
  `test_results/thesis_benchmark_cluster_large_n100_d1_20260426`.
  Paper-ready tables and figures are under:
  `test_results/thesis_benchmark_cluster_large_n100_d1_20260426/paper_artifacts`.
  Average gaps to LKH:
  - baseline: `30.023362228761982%`
  - new full learnable bias: `19.64153639788954%`
  - scheduled/fixed bias: `17.806783359242594%`
  - true `w/o all bias`: `20.34274308077898%`
  - `w/o fusion gate`: `23.349276883571527%`
  - `w/o group embedding`: `16.655800247413705%`
  All six benchmark evaluations had `feasible_rate = 1.0`.
- 2026-04-26: Replaced the earlier unaugmented six-model synthetic and benchmark
  test outputs after discovering that `scripts/evaluate_ctspd.py` had not been
  passing `aug_factor` to the environment. The committed outputs in
  `test_results/thesis_main_synthetic_n100_g8_d1_20260426` and
  `test_results/thesis_benchmark_cluster_large_n100_d1_20260426` now reflect
  augmented inference.
- 2026-04-26: Completed reproducibility check and saved the report under:
  `test_results/reproducibility_check_20260426`.
  The check passed. It included strict CPU loading of all six
  `checkpoint-best.pt` files, metadata/row-count/seed/feasibility validation
  for stored synthetic and benchmark outputs, PNG/PDF artifact validation, and
  an exact dynamic rerun comparison of the six-model external benchmark.
- 2026-04-26: Added same-priority local search and tour saving to
  `scripts/evaluate_ctspd.py`, then ran the six final models on the external
  LKH benchmark with the full enhanced inference budget:
  `anchor,mds` features, 8-fold augmentation, greedy plus 64 stochastic
  sampling runs, and same-priority local search with up to 20 passes. Results
  are saved under
  `test_results/thesis_benchmark_cluster_large_n100_d1_aug8_sample64_ls20_20260426`.
  The enhanced external benchmark improves all models but still does not make
  the new full learnable-bias model the best external-benchmark model:
  `w/o group embedding` has average gap `6.88054973950998%`, scheduled/fixed
  bias has `11.509764257512131%`, and full learnable bias has
  `13.571954288993018%`.
  A non-committed local diagnostic run using the full learnable checkpoint
  while disabling group embedding or the fusion gate at inference time did not
  improve the full checkpoint on the external benchmark: the average gaps were
  `18.1056477089938%` and `20.677089255168337%`, respectively.
- 2026-04-26: Added model-only test-time sampling controls to
  `scripts/evaluate_ctspd.py` and the model forward paths:
  `sampling_temperature`, `sampling_top_k`, plus extra benchmark feature modes
  `farthest`, `central`, and `meanmax`. These controls are used only for
  softmax sampling inference and do not change model weights.
  Completed full learnable-bias model-only inference explorations are saved
  under:
  `test_results/full_learnable_model_only_temperature_sweep_20260426`,
  `test_results/full_learnable_model_only_parallel_ensemble_20260426`,
  `test_results/full_learnable_model_only_mds_deep_sampling_20260426`, and the
  consolidated best result
  `test_results/full_learnable_model_only_best_of_ensemble_20260426`.
  The consolidated full-only ensemble average cost is `26574.1` with average
  LKH gap `17.206078282859504%` and feasible rate `1.0`. This improves over
  the standard full learnable external benchmark result (`27134.5`,
  `19.64153639788954%`) and slightly beats the scheduled/fixed model under the
  standard benchmark protocol (`26742.2`, `17.806783359242594%`), but still does
  not beat `w/o group embedding` under the standard protocol (`26507.0`,
  `16.655800247413705%`). This model-only result also does not supersede the
  stronger full learnable enhanced result (`25761.3`,
  `13.571954288993018%`), because that enhanced result used sampling64 plus
  same-priority local search. A later targeted sample512 run was started and then
  stopped at user request; its partial outputs were discarded and are not
  committed.
- 2026-04-27: Added LKH 3.0.14 as a classical synthetic benchmark. LKH's
  upstream CTSP-D convention did not match this repository's priority rule, so
  `LKH-3.0.14/SRC/Penalty_CTSP_D.c` now supports a compile-time
  `CTSPD_LOW_PRIORITY_FIRST` mode and `LKH-3.0.14/SRC/Makefile` supports:
  `make CTSPD_PRIORITY=LOW_FIRST`. The default build without this flag remains
  the upstream behavior.
- 2026-04-27: Added `scripts/run_lkh_ctspd_benchmark.py` to export the fixed
  synthetic `.pt` dataset to LKH `TYPE : CTSP-D` instances with explicit
  distance matrices, run the patched LKH, normalize tours by rotation/reversal,
  and write per-instance costs/times.
- 2026-04-27: Deleted the previous main synthetic result directory
  `test_results/thesis_main_synthetic_n100_g8_d1_20260426` to avoid confusion,
  reran all six neural models on
  `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt` with
  `--augmentation-factor 8`, and ran LOW_FIRST LKH on all 1000 instances with
  `MAX_TRIALS=200`, `RUNS=1`.
- 2026-04-27: Added `scripts/make_main_lkh_artifacts.py` and generated the
  unified main result directory:
  `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427`.
  LKH summary:
  - average cost = `15.132940`
  - feasible rate = `1.0`
  - total wall-clock time = `7189.209` seconds
  - time per instance = `7.189` seconds
  The new full learnable-bias model remains the best neural model:
  average cost = `15.445869`, average gap to LKH = `2.087%`, time per
  instance = `0.069` seconds, about `104.7x` faster than LKH.
- 2026-04-27: Generated paper artifacts under
  `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts`,
  including the main table, ablation table, per-instance costs, pairwise win
  tables, average-cost bar chart, gap-to-LKH chart, per-instance gap boxplot,
  pairwise win heatmap, inference-time log-scale bar chart, and training-score
  curves.
- 2026-04-27: Cleaned the project structure for final thesis submission.
  Retained the core CTSP-d code, final `checkpoint-best.pt` files, fixed
  synthetic dataset, final LKH-backed result directory, and final paper figures.
  Moved non-final external benchmark outputs, exploratory inference outputs,
  original TSP reference code, external benchmark package, `checkpoint-latest.pt`
  files, per-epoch result images, and runtime logs into local ignored
  `archive_to_review/`. Added `PROJECT_STRUCTURE.md`, `MODEL_ARTIFACTS.md`,
  `RESULT_ARTIFACTS.md`, and `PROJECT_FILE_AUDIT.md`.

Update this section whenever long-running training or evaluation jobs are
started, stopped, or completed.
