# Project File Audit

Updated: 2026-04-27

This audit was created before cleanup. Initial checks:

- Git branch: `main`
- Working tree before cleanup: clean, except ignored runtime/cache files
- Cleanup principle: keep reproducible thesis code, final checkpoints, final synthetic test data, final LKH-backed results, and final paper figures. Archive exploratory or ambiguous material locally under `archive_to_review/` rather than deleting it.

## Must Keep: Reproduction Code

- Repository overview and handoff:
  - `README.md`
  - `EXPERIMENT_HANDOFF.md`
  - `requirements.txt`
  - `pyrightconfig.json`
- Baseline CTSP-d code:
  - `CSTPd_bsl/CTSPd_ProblemDef.py`
  - `CSTPd_bsl/POMO/CTSPd_Env.py`
  - `CSTPd_bsl/POMO/CTSPd_Model.py`
  - `CSTPd_bsl/POMO/CTSPd_Trainer.py`
  - `CSTPd_bsl/POMO/CTSPd_Tester.py`
  - `CSTPd_bsl/POMO/train_n100.py`
- Priority/group-aware CTSP-d code:
  - `CSTPd_cluster/CTSPd_ProblemDef.py`
  - `CSTPd_cluster/POMO/CTSPd_Env.py`
  - `CSTPd_cluster/POMO/CTSPd_Model.py`
  - `CSTPd_cluster/POMO/CTSPd_Trainer.py`
  - `CSTPd_cluster/POMO/CTSPd_Tester.py`
  - `CSTPd_cluster/POMO/train_n100.py`
  - `CSTPd_cluster/POMO/train_n100_learnable_bias.py`
  - `CSTPd_cluster/POMO/train_n100_wo_all_bias.py`
  - `CSTPd_cluster/POMO/train_n100_wo_fusion_gate.py`
  - `CSTPd_cluster/POMO/train_n100_wo_group_embedding.py`
- Final utility scripts:
  - `scripts/generate_synthetic_test_dataset.py`
  - `scripts/evaluate_ctspd.py`
  - `scripts/summarize_results.py`
  - `scripts/run_training_queue.py`
  - `scripts/run_lkh_ctspd_benchmark.py`
  - `scripts/make_main_lkh_artifacts.py`
  - `scripts/plot_paper_figures.py`
- LKH benchmark implementation:
  - `LKH-3.0.14/`
  - Keep source and docs. Do not commit compiled binary/object files.

## Must Keep: Final Models and Checkpoints

The thesis main results use `checkpoint-best.pt`, not `checkpoint-latest.pt`.

| Model | Final checkpoint |
|---|---|
| POMO baseline | `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/checkpoint-best.pt` |
| Full model | `CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias/checkpoint-best.pt` |
| scheduled bias | `CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1/checkpoint-best.pt` |
| w/o all bias | `CSTPd_cluster/POMO/result/26日_10点20分_cluster_n100_d1_wo_all_bias/checkpoint-best.pt` |
| w/o fusion gate | `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1/checkpoint-best.pt` |
| w/o group emb. | `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1/checkpoint-best.pt` |

Must also keep, per model:

- `training_metrics.csv`
- `training_progress.json`
- `latest-train_score.jpg`
- `latest-train_loss.jpg`
- result-folder `src/` snapshots, because they record the training-time code/config state.

## Must Keep: Final Test Dataset

- `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt`
- `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.json`

## Must Keep: Final Experiment Results

Main thesis result directory:

- `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/`

Key files include:

- `summary.csv`
- `evaluations/*/test_instances.csv`
- `evaluations/*/test_summary.csv`
- `evaluations/*/test_summary.json`
- `evaluations/lkh_low_first/lkh_instances.csv`
- `evaluations/lkh_low_first/lkh_summary.json`
- LKH generated `.ctspd` instances, `.par` files, logs, raw tours, and normalized tours. These are retained because they make the LKH benchmark auditable.

## Must Keep: Final Paper Figures

Final figure directory:

- `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts/`

Retain:

- final `.pdf` and `.png` figures
- `FIGURE_HANDOFF.md`
- `FIGURE_MANIFEST.csv`
- `README.md`
- source CSV/Markdown tables used to generate figures
- `route_case_study_summary.json`

## Directly Deletable Cache/System Files

Safe to delete:

- `__pycache__/`
- `.ipynb_checkpoints/`
- `.DS_Store`
- `Thumbs.db`
- compiled LKH outputs already ignored by Git:
  - `LKH-3.0.14/LKH`
  - `LKH-3.0.14/SRC/OBJ/`

## Archive: Intermediate Test Results

Move to local `archive_to_review/old_results/` and remove from the final Git-tracked project:

- `test_results/full_learnable_model_only_temperature_sweep_20260426/`
- `test_results/full_learnable_model_only_parallel_ensemble_20260426/`
- `test_results/full_learnable_model_only_mds_deep_sampling_20260426/`
- `test_results/full_learnable_model_only_best_of_ensemble_20260426/`
- `test_results/reproducibility_check_20260426/`
- `test_results/thesis_benchmark_cluster_large_n100_d1_20260426/`
- `test_results/thesis_benchmark_cluster_large_n100_d1_aug8_sample64_ls20_20260426/`
- `comparison_results/`

Rationale: these are external benchmark, enhanced inference, model-only ensemble, or historical reproducibility outputs. They are useful context but not part of the final LKH-backed synthetic main experiment.

## Archive: Intermediate/External Data

Move to local `archive_to_review/external_benchmarks/`:

- `CTSPd(SOTA)/`

Rationale: this supports the older external benchmark checks, but the final thesis main experiment uses the fixed synthetic dataset and the LOW_FIRST LKH run under `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/`.

## Archive: Non-final Code

Move to local `archive_to_review/non_final_code/`:

- `TSP/`
- `scripts/compare_sota_instances.py`
- `scripts/postprocess_same_priority_local_search.py`

Rationale: these are useful reference/external-benchmark utilities, but they are not required for reproducing the final synthetic thesis experiment.

## Archive: Non-final Model Artifacts

Move to local `archive_to_review/non_final_model_artifacts/`:

- all six `checkpoint-latest.pt` files
- per-epoch image folders under final result directories:
  - `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/img/`
  - `CSTPd_cluster/POMO/result/*/img/`
- runtime `.out` logs under model result folders

Rationale: the thesis uses `checkpoint-best.pt`; `training_metrics.csv`, `training_progress.json`, and latest score/loss images are retained.

## Cleanup Actions Applied

- Created local ignored `archive_to_review/`.
- Moved old `test_results/` experiment folders and `comparison_results/` into
  `archive_to_review/old_results/`.
- Moved `CTSPd(SOTA)/` into
  `archive_to_review/external_benchmarks/`.
- Moved `TSP/` and the two non-final external-benchmark helper scripts into
  `archive_to_review/non_final_code/`.
- Moved `checkpoint-latest.pt`, per-epoch `img/` folders, and runtime `.out`
  logs into `archive_to_review/non_final_model_artifacts/`.
- Deleted cache/build artifacts: `__pycache__/`, `.ipynb_checkpoints/`,
  `LKH-3.0.14/LKH`, and `LKH-3.0.14/SRC/OBJ/`.

## Needs Human Confirmation

Kept in place for now:

- `CSTPd_bsl/POMO/train_n20.py`, `train_n50.py`, `test_n20.py`, `test_n100.py`, `test.py`
- `CSTPd_cluster/POMO/train_n20.py`, `train_n50.py`, `test_n20.py`, `test_n100.py`, `test.py`
- `CSTPd_cluster/POMO/train_n100_wo_cluster_bias.py`
- `CSTPd_cluster/POMO/train_n100_wo_priority_distance_bias.py`
- `utils/`

Rationale: these files are not central to the final thesis result, but they are small and may still be useful for debugging, smoke tests, or describing ablation history. They are not moved automatically.
