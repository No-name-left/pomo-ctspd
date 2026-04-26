# Main Synthetic Test Artifacts: n100 g8 d1

This folder contains paper-oriented outputs for the final six-model same-distribution synthetic CTSP-d evaluation.

Inference setting: 8-fold geometric augmentation, greedy/POMO decoding, and best candidate selection across augmented directions and POMO starts. No stochastic sampling is used for this main synthetic ablation test.

Files:
- `main_results_table.csv` and `.md`: average cost, confidence intervals, feasibility, training metadata, and deltas.
- `ablation_table.csv`: ablation deltas against the full learnable-bias model.
- `pairwise_win_table.csv`: instance-level wins/losses against baseline and full model.
- `test_average_cost_dotplot.png/.pdf`, `improvement_vs_baseline_bar.png/.pdf`, `ablation_delta_vs_full_bar.png/.pdf`, and `training_score_curves.png/.pdf`: paper-ready figures.
