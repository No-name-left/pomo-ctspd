# Result Artifacts

Updated: 2026-04-27

This file records the final data and result files retained for the thesis main
experiment.

## Final Dataset

| File | Purpose | Used in thesis |
|---|---|---|
| `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt` | Fixed 1000-instance synthetic CTSP-d test set with `n=100`, `num_groups=8`, `d=1`, seed `20260423`. | Yes |
| `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.json` | Metadata sidecar for the fixed test set. | Yes |

## Final Main Result Directory

Main result root:
`test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/`

| Result file or directory | Model / source | Dataset | Metrics recorded | Used in thesis | Related figures |
|---|---|---|---|---|---|
| `summary.csv` | LKH plus all six neural methods | Fixed synthetic test set | Average cost, gap to LKH, gap to baseline/full model, feasibility, inference time, speedup, checkpoint path | Yes | Main tables; cost/gap/time/tradeoff/ablation figures |
| `evaluations/baseline/test_instances.csv` and `test_summary.*` | POMO baseline | Fixed synthetic test set | Per-instance cost/feasibility/time and aggregate summary | Yes | Cost, gap, pairwise, delta, route-case support |
| `evaluations/full_learnable/test_instances.csv` and `test_summary.*` | Full model | Fixed synthetic test set | Per-instance cost/feasibility/time and aggregate summary | Yes | Cost, gap, pairwise, delta, ablation, route-case support |
| `evaluations/scheduled_bias/test_instances.csv` and `test_summary.*` | scheduled bias | Fixed synthetic test set | Per-instance cost/feasibility/time and aggregate summary | Yes | Cost, gap, pairwise, ablation, route-case support |
| `evaluations/wo_all_bias/test_instances.csv` and `test_summary.*` | w/o all bias | Fixed synthetic test set | Per-instance cost/feasibility/time and aggregate summary | Yes | Cost, gap, pairwise, ablation, route-case support |
| `evaluations/wo_fusion_gate/test_instances.csv` and `test_summary.*` | w/o fusion gate | Fixed synthetic test set | Per-instance cost/feasibility/time and aggregate summary | Yes | Cost, gap, pairwise, ablation, route-case support |
| `evaluations/wo_group_embedding/test_instances.csv` and `test_summary.*` | w/o group emb. | Fixed synthetic test set | Per-instance cost/feasibility/time and aggregate summary | Yes | Cost, gap, pairwise, ablation, route-case support |
| `evaluations/lkh_low_first/lkh_instances.csv` and `lkh_summary.json` | LOW_FIRST-patched LKH | Fixed synthetic test set exported to CTSP-D instances | Per-instance LKH tour length/time/feasibility and aggregate summary | Yes | LKH reference for all gap/time figures |
| `evaluations/lkh_low_first/instances/`, `params/`, `raw_tours/`, `normalized_tours/`, `logs/` | LOW_FIRST-patched LKH | Fixed synthetic test set | Generated LKH inputs, parameters, raw outputs, normalized tours, logs | Yes, source data | Route case study and LKH reproducibility |

## Final Paper Artifacts

Paper artifact root:
`test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts/`

| File | Purpose | Data source | Thesis placement |
|---|---|---|---|
| `main_results_table.csv/.md` | Main comparison table | `summary.csv` | Main experiment |
| `ablation_table.csv/.md` | Ablation comparison table | `summary.csv` | Ablation analysis |
| `per_instance_costs.csv` | Per-instance merged costs and gaps | Final per-model test results plus LKH | Source for distribution/delta/route figures |
| `pairwise_win_counts.csv`, `pairwise_win_percent.csv` | Pairwise per-instance win rates | `per_instance_costs.csv` | Pairwise-win heatmap |
| `FIGURE_MANIFEST.csv` | Figure file inventory | Plot scripts | Artifact management |
| `FIGURE_HANDOFF.md` | Figure purpose, placement, captions, and downgrade/delete decisions | Final figures and source CSVs | Thesis writing handoff |
| `MISSING_FIGURES.md` | Reasons for not generating unsupported suggested figures | Available final data | Handoff note |

The final figure set includes PDF and PNG versions for:

- `average_cost_with_lkh_bar`
- `gap_to_lkh_bar`
- `per_instance_gap_to_lkh_boxplot`
- `pairwise_win_heatmap`
- `time_per_instance_bar_log`
- `quality_time_tradeoff_scatter`
- `paired_gap_delta_to_baseline`
- `ablation_summary_delta`
- `route_case_study_panels`
- `route_case_group_sequence_appendix`
- `training_score_curves`
- `training_score_curves_appendix_all`

## Archived or Removed from Final Result Set

The previous external-benchmark and exploratory result directories were moved
to local `archive_to_review/old_results/` and are not part of the final
submitted result set. They remain available for manual inspection on this
machine but should not be treated as thesis-main reproduction artifacts unless
the thesis scope is deliberately expanded.
