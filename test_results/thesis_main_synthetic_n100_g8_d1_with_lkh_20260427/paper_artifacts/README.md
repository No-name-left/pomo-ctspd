# Paper Artifacts

This folder contains thesis-ready figures, tables, and handoff notes for the synthetic CTSP-d main experiment with LKH.

Use the `.pdf` files in the thesis whenever possible. The `.png` files are 300 dpi previews.

Key files:

- `FIGURE_HANDOFF.md`: detailed description, recommended placement, and captions for each figure.
- `FIGURE_MANIFEST.csv`: machine-readable figure inventory.
- `main_results_table.csv` / `.md`: main result table.
- `ablation_table.csv` / `.md`: ablation table.
- `per_instance_costs.csv`: aligned per-instance costs and gaps.
- `pairwise_win_counts.csv` / `pairwise_win_percent.csv`: pairwise comparison tables.

Generated figures:

- `average_cost_with_lkh_bar.pdf/png`: 展示 LKH 与各神经模型的平均路径成本。
- `gap_to_lkh_bar.pdf/png`: 展示各模型相对 LKH 的平均 gap。
- `per_instance_gap_to_lkh_boxplot.pdf/png`: 展示每个模型在逐实例层面的 gap 分布稳定性。
- `pairwise_win_heatmap.pdf/png`: 展示模型之间逐实例两两比较的胜率。
- `time_per_instance_bar_log.pdf/png`: 展示 LKH 与神经模型的平均单实例推理时间。
- `training_score_curves.pdf/png`: 展示主要模型训练 score 曲线。
- `training_score_curves_appendix_all.pdf/png`: 展示全部神经模型训练 score 曲线。
- `quality_time_tradeoff_scatter.pdf/png`: 展示解质量与推理时间之间的折中关系。
- `paired_gap_delta_to_baseline.pdf/png`: 展示每个模型相对 baseline 的逐实例 gap 改变量。
- `ablation_summary_delta.pdf/png`: 直接展示各消融模型相对 Full model 的平均 gap 变化。
- `feasibility_rate_bar.pdf/png`: 展示各方法生成解的 CTSP-d 可行率。
- `route_case_study_panels.pdf/png`: 展示同一代表性实例上不同模型路线形态的差异。
- `route_case_group_sequence.pdf/png`: 展示同一实例中不同模型访问 priority group 的顺序差异。
