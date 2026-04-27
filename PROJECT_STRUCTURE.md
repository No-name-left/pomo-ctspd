# Project Structure

Updated: 2026-04-27

This repository is organized around the final reproducible thesis experiment for
POMO-based CTSP-d solving with a LOW_FIRST LKH benchmark.

## Main Directories

| Path | Role | Keep reason |
|---|---|---|
| `CSTPd_bsl/` | CTSP-d POMO baseline implementation | Baseline model, trainer, tester, environment, final baseline checkpoint and logs. |
| `CSTPd_cluster/` | Priority-group-aware CTSP-d implementation | Full model, scheduled-bias variant, and ablation model code/checkpoints. |
| `scripts/` | Reproduction and reporting scripts | Dataset generation, checkpoint evaluation, LKH benchmark, result aggregation, and paper figure generation. |
| `data/synthetic_tests/` | Fixed synthetic test data | Contains the final `n=100`, `g=8`, `d=1`, 1000-instance test set used in the thesis main experiment. |
| `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/` | Final main experiment results | Contains the six neural-model evaluations, LKH reference results, combined `summary.csv`, and final paper figures. |
| `LKH-3.0.14/` | LOW_FIRST-patched LKH source | Classical heuristic benchmark source. Build locally with `make CTSPD_PRIORITY=LOW_FIRST`. Build outputs are ignored. |
| `utils/` | Shared utility code | Logging/result helper utilities used by training scripts; retained for compatibility. |
| `archive_to_review/` | Local ignored archive | Contains non-final experiments, old result snapshots, external benchmark data, and intermediate artifacts that were not deleted outright. Not intended for Git submission. |

## Final Reproduction Flow

1. Install dependencies from `requirements.txt`.
2. Use the fixed dataset already stored at:
   `data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt`.
3. Evaluate neural checkpoints with `scripts/evaluate_ctspd.py`.
4. Build the LKH binary when needed:
   `make -C LKH-3.0.14 CTSPD_PRIORITY=LOW_FIRST`.
5. Run or reuse the LKH results under:
   `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/lkh_low_first/`.
6. Merge results and regenerate tables/figures with:
   `scripts/make_main_lkh_artifacts.py` and `scripts/plot_paper_figures.py`.

## Final Artifact Indexes

- Model/checkpoint inventory: `MODEL_ARTIFACTS.md`
- Result/data inventory: `RESULT_ARTIFACTS.md`
- Figure handoff:
  `test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts/FIGURE_HANDOFF.md`
- File audit and cleanup decisions: `PROJECT_FILE_AUDIT.md`

## Archived Content

The cleanup moved non-final or uncertain content into local
`archive_to_review/`, preserving source paths where practical. This includes:

- Old external benchmark result folders from `test_results/`.
- Earlier model-only inference explorations.
- Historical `comparison_results/`.
- The external `CTSPd(SOTA)/` benchmark package.
- The original `TSP/` POMO reference folder.
- Non-final `checkpoint-latest.pt` files.
- Per-epoch training curve images under result `img/` folders.
- Runtime `.out` logs.

These files are ignored by Git. If a future write-up needs any of them, inspect
`archive_to_review/` manually before deleting it.
