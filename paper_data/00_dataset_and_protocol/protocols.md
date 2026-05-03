# Experiment Protocol Notes

## Final Main Synthetic Experiment

- Directory: `paper_data/01_main_synthetic_n100_g8_d1_lkh/`
- Dataset: `synthetic_n100_g8_d1_1000_seed20260423.pt`
- Dataset metadata: `synthetic_n100_g8_d1_1000_seed20260423.json`
- Instance count: 1000
- Problem size: 100
- Priority groups: 8
- Relaxation level: `d = 1`
- Seed: 20260423
- Neural inference: 8-fold geometric augmentation, greedy/POMO best selection.
- LKH reference: LOW_FIRST priority convention, max trials 200, one run,
  scale 1,000,000.
- Main paper table: `01_main_synthetic_n100_g8_d1_lkh/paper_tables_and_case_studies/main_results_table.md`
- Aligned per-instance costs: `01_main_synthetic_n100_g8_d1_lkh/paper_tables_and_case_studies/per_instance_costs.csv`

## Recovered External Benchmark: Aug8

- Directory: `paper_data/02_external_cluster_large_n100_d1_aug8/`
- Source commit: `1e6ca97`
- Dataset: CTSPd_SOTA `Cluster_large` n100/d1 subset.
- Instance count: 10
- Instance/LKH reference list: `external_cluster_large_n100_d1_lkh_reference.csv`
- Reconstructed feature modes: `anchor,mds`
- Neural inference: 8-fold augmentation, greedy/POMO best selection by the
  original CTSP-d distance matrix.
- Purpose: historical external/generalization check, not the final
  same-distribution thesis result.

## Recovered External Benchmark: Aug8 + Sample64 + LS20

- Directory: `paper_data/03_external_cluster_large_n100_d1_aug8_sample64_ls20/`
- Source commit: `3d64d7e`
- Dataset: same 10 CTSPd_SOTA `Cluster_large` n100/d1 instances.
- Reconstructed feature modes: `anchor,mds`
- Neural inference: 8-fold augmentation, 64 stochastic sampling runs, and
  same-priority local search with 20 passes.
- Purpose: enhanced inference stress test for the historical external benchmark.

## Model Set

The six final neural variants are:

- `baseline`: POMO baseline.
- `full_learnable`: cluster-aware full model with learnable bias.
- `scheduled_bias`: cluster-aware scheduled/fixed bias.
- `wo_all_bias`: cluster-aware model without attention bias.
- `wo_fusion_gate`: structural ablation without fusion gate.
- `wo_group_embedding`: structural ablation without group embedding.

Training curves and checkpoint metadata are summarized in
`04_training_runs/training_runs_manifest.csv`.
