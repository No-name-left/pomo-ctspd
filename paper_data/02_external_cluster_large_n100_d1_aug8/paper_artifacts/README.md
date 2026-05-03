# External Benchmark Artifacts: Cluster_large n100 d1

This folder contains paper-oriented outputs for the external CTSP-d benchmark check on 10 `Cluster_large` instances with 100 nodes and relaxation level d=1.

Inference setting: two reconstructed feature modes (`anchor` and `mds`), 8-fold geometric augmentation on those features, greedy/POMO decoding, and best candidate selection by the original CTSP-d distance matrix. No stochastic sampling or local search is used in this six-model table.

Inputs:
- Instances: `CTSPd(SOTA)/INSTANCES/Cluster_large/*100-C-*-1-*.ctspd`
- LKH references: parsed from `CTSPd(SOTA)/TOURS/Cluster_large/*.tour` into `../lkh_reference.csv`
- Models: the final six thesis checkpoints listed in `EXPERIMENT_HANDOFF.md`

Files:
- `benchmark_results_table.csv` and `.md`: average cost, gap to LKH, feasibility, and baseline deltas.
- `per_instance_costs.csv`: LKH cost plus every model cost/gap and selected feature mode for each instance.
- `per_instance_winner_counts.csv`: lower-cost model wins over the 10 benchmark instances.
- `pairwise_win_table.csv`: pairwise wins/losses between models on the same 10 instances.
- `gap_to_lkh_bar.png/.pdf`, `per_instance_gap_heatmap.png/.pdf`, and `model_costs_vs_lkh.png/.pdf`: paper-ready figures.

Interpretation note:
This benchmark is an external/generalization check. The models were trained on the thesis synthetic n100/g8/d1 distribution, not on these TSPLIB-derived benchmark instances. Use the 1000-instance fixed synthetic test as the main ablation evidence, and use this folder as a transparent LKH comparison and qualitative stress test.
