# Model Artifacts

Updated: 2026-04-27

This file lists the final model artifacts retained for the thesis main
experiment. The submitted model artifact for each neural method is
`checkpoint-best.pt`; non-final `checkpoint-latest.pt` files were moved to
`archive_to_review/`.

| Model | Checkpoint | Config / source record | Training log | Test result | Used in thesis main result | Notes |
|---|---|---|---|---|---|---|
| POMO baseline | `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/checkpoint-best.pt` | `CSTPd_bsl/POMO/train_n100.py`; source snapshot in `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/src/` | `CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/training_metrics.csv`; `training_progress.json`; latest curve images | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/baseline/` | Yes | Baseline embeds `(x, y, priority)` directly and uses CTSP-d masking in the environment. |
| Full model | `CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias/checkpoint-best.pt` | `CSTPd_cluster/POMO/train_n100_learnable_bias.py`; source snapshot in result `src/` | `training_metrics.csv`; `training_progress.json`; latest curve images in the same result folder | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/full_learnable/` | Yes | Priority/group-aware model with group embedding, fusion gate, scheduled residual same-group bias, and learnable relation bias. |
| scheduled bias | `CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1/checkpoint-best.pt` | `CSTPd_cluster/POMO/train_n100.py`; source snapshot in result `src/` | `training_metrics.csv`; `training_progress.json`; latest curve images in the same result folder | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/scheduled_bias/` | Yes | Uses hand-crafted scheduled/fixed priority-group bias; interpreted as the scheduled-bias ablation. |
| w/o all bias | `CSTPd_cluster/POMO/result/26日_10点20分_cluster_n100_d1_wo_all_bias/checkpoint-best.pt` | `CSTPd_cluster/POMO/train_n100_wo_all_bias.py`; source snapshot in result `src/` | `training_metrics.csv`; `training_progress.json`; latest curve images in the same result folder | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/wo_all_bias/` | Yes | Keeps group embedding/fusion but removes same-group, priority-distance, relation, and decoder priority bias terms. |
| w/o fusion gate | `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1/checkpoint-best.pt` | `CSTPd_cluster/POMO/train_n100_wo_fusion_gate.py`; source snapshot in result `src/` | `training_metrics.csv`; `training_progress.json`; latest curve images in the same result folder | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/wo_fusion_gate/` | Yes | Legacy scheduled-bias structural ablation; removes the coordinate/group fusion gate. |
| w/o group emb. | `CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1/checkpoint-best.pt` | `CSTPd_cluster/POMO/train_n100_wo_group_embedding.py`; source snapshot in result `src/` | `training_metrics.csv`; `training_progress.json`; latest curve images in the same result folder | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/wo_group_embedding/` | Yes | Legacy scheduled-bias structural ablation; removes explicit group embedding. |
| LOW_FIRST LKH | No checkpoint | `LKH-3.0.14/` source; build with `make CTSPD_PRIORITY=LOW_FIRST` | LKH logs under final result directory | `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/lkh_low_first/` | Yes | Classical heuristic reference, not a neural checkpoint. |

## Size / Git Notes

The retained neural `checkpoint-best.pt` files are final thesis artifacts. If
the remote repository has strict binary-size limits, migrate these checkpoint
files to Git LFS or document an external artifact download path before pushing.
