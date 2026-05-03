# Thesis Experiment Data Index

This directory is the paper-facing data bundle for the CTSP-d thesis experiments.
It is organized for direct reading by humans and AI tools. Costs are tour lengths;
lower is better. Gap columns are percentages relative to the corresponding LKH
reference unless the filename says otherwise.

## Directory Map

| Directory | Purpose | Primary files |
| --- | --- | --- |
| `00_dataset_and_protocol/` | Dataset metadata and protocol notes. | `protocols.md`, synthetic dataset `.json`/`.pt`, external LKH reference CSV |
| `01_main_synthetic_n100_g8_d1_lkh/` | Final same-distribution thesis experiment on 1000 fixed synthetic n100/g8/d1 instances with LOW_FIRST LKH. | `summary.csv`, `paper_tables_and_case_studies/main_results_table.*`, `per_instance_costs.csv`, per-model `evaluations/*/test_instances.csv` |
| `02_external_cluster_large_n100_d1_aug8/` | Recovered historical external benchmark on 10 CTSPd_SOTA Cluster_large n100/d1 instances. | `summary.csv`, `lkh_reference.csv`, `paper_artifacts/per_instance_costs.csv` |
| `03_external_cluster_large_n100_d1_aug8_sample64_ls20/` | Recovered enhanced external benchmark with sampling and same-priority local search. | `summary.csv`, `per_instance_costs.csv`, `pairwise_win_table.csv`, generated `.tour` files |
| `04_training_runs/` | Training metrics and progress for the six final model variants. | `training_runs_manifest.csv`, per-model `training_metrics.csv`, `training_progress.json` |

`EXPERIMENT_INDEX.csv` gives a compact machine-readable overview of the
experiment groups. `MANIFEST.csv` lists the bundled data files, excluding the
manifest itself, and is regenerated after the data bundle is updated.

## Provenance

- Main synthetic result source:
  `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/`.
- External aug8 result recovered from git commit `1e6ca97`
  (`Add final augmented CTSP-d thesis evaluations`).
- External aug8 + sample64 + LS20 result recovered from git commit `3d64d7e`
  (`Add enhanced LKH benchmark inference results`).

Images and figure styling were not regenerated in this cleanup. Existing figure
artifacts remain in their source result folders; the curated main synthetic
bundle copies only CSV/JSON/Markdown data needed for paper tables and analysis.
