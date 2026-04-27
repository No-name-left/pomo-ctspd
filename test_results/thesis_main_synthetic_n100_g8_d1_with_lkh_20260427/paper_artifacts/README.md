# Paper Artifacts

Generated from `/root/autodl-tmp/test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427` using `make_main_lkh_artifacts.py`.

Common instances across all methods: 1000.

Files:
- `main_results_table.csv` / `.md`: compact table for the main synthetic experiment.
- `ablation_table.csv` / `.md`: neural-model ablation table using the full model as reference.
- `per_instance_costs.csv`: aligned per-instance costs and gaps.
- `pairwise_win_counts.csv` and `pairwise_win_percent.csv`: pairwise comparison tables.
- `average_cost_with_lkh_bar.*`: average cost with 95% confidence intervals.
- `gap_to_lkh_bar.*`: average per-instance gap to LKH.
- `per_instance_gap_to_lkh_boxplot.*`: distribution of per-instance gaps to LKH.
- `pairwise_win_heatmap.*`: pairwise win-rate heatmap.
- `time_per_instance_bar_log.*`: inference-time comparison on a log scale.
- `training_score_curves.*`: training-score curves for neural models.

LKH here refers to the LOW_FIRST-patched CTSP-d build used for the synthetic data convention.
