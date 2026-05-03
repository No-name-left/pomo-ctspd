# POMO for CTSP-d

基于 POMO 的 CTSP-d 求解实验代码。项目将原始 TSP/POMO 框架扩展到 **Clustered Traveling Salesman Problem with d-relaxed priority rule (CTSP-d)**，并提供 baseline 与 cluster-aware 两套模型实现。

CTSP-d 中，每个节点包含坐标和优先级组：`(x, y, priority)`。在构造路径时，模型必须满足 d-relaxed priority rule：若当前未访问节点中的最高优先级为 `p`，下一步只能访问优先级位于 `[p, p + d]` 的节点。代码里优先级数值越小，优先级越高。

## Highlights

- CTSP-d environment with priority-aware action masking.
- POMO rollout and policy-gradient training.
- Random CTSP-d instance generation with priority groups.
- 8-fold geometric augmentation for random coordinate instances.
- `.ctspd` benchmark parser with real-distance evaluation.
- Baseline model and cluster-aware model for comparison.
- LOW_FIRST-patched LKH runner for synthetic CTSP-d benchmark comparison.

## Repository Layout

```text
CSTPd_bsl/         CTSP-d POMO baseline code and final baseline artifacts
CSTPd_cluster/     Priority/group-aware CTSP-d code, full model, and ablations
data/              Fixed synthetic test set used by the thesis main experiment
paper_data/        Curated thesis data bundle with recovered external results
scripts/           Dataset generation, evaluation, LKH benchmark, and figures
test_results/      Final LKH-backed synthetic experiment results
LKH-3.0.14/        LKH source used for the LOW_FIRST synthetic benchmark
utils/             Logging, result folders, and training-curve utilities
archive_to_review/ Local ignored archive for non-final or uncertain files
```

Detailed artifact indexes are maintained in `PROJECT_STRUCTURE.md`,
`MODEL_ARTIFACTS.md`, `RESULT_ARTIFACTS.md`, `paper_data/README.md`, and
`test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/paper_artifacts/FIGURE_HANDOFF.md`.

## Model Variants

**Baseline**: `CSTPd_bsl/POMO/`

The baseline model feeds `[x, y, priority]` directly into the node embedding layer. The CTSP-d constraint is enforced in the environment through the priority mask.

**Cluster-aware**: `CSTPd_cluster/POMO/`

The cluster-aware model embeds coordinates and priority groups separately, then fuses them before the transformer encoder. Its encoder also adds a same-group attention bias so the model can explicitly use priority-group structure.

## Quick Start

Install dependencies:

```bash
conda create -n py310-env python=3.10
conda activate py310-env
pip install torch numpy matplotlib
```

The final thesis setting is `n=100`, `num_groups=8`, `d=1`. The fixed test set
is already included at:

```text
data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt
```

Generate the same test set again if needed:

```bash
python scripts/generate_synthetic_test_dataset.py \
  --problem-size 100 \
  --num-groups 8 \
  --relaxation-d 1 \
  --instance-num 1000 \
  --seed 20260423
```

Evaluate a final checkpoint on the fixed synthetic test set:

```bash
python scripts/evaluate_ctspd.py \
  --model-type cluster \
  --model-variant learnable_bias \
  --checkpoint "CSTPd_cluster/POMO/result/<full_learnable_run>/checkpoint-best.pt" \
  --mode synthetic \
  --dataset-file data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt \
  --augmentation-factor 8
```

Regenerate final paper figures from the stored results:

```bash
python scripts/plot_paper_figures.py \
  --result-dir test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427
```

Legacy small-size training scripts such as `train_n20.py` and `train_n50.py`
remain for smoke tests, but they are not part of the final thesis main result.

## Paper Data Bundle

The paper-facing data bundle is:

```text
paper_data/
```

It contains the fixed dataset metadata, curated main synthetic results,
recovered historical external benchmark results, aligned per-instance tables,
pairwise/winner tables, and final training metrics. Start from:

- `paper_data/README.md`
- `paper_data/EXPERIMENT_INDEX.csv`
- `paper_data/MANIFEST.csv`

The historical external benchmark results were recovered from git history and
renamed for clarity:

- `paper_data/02_external_cluster_large_n100_d1_aug8/`
- `paper_data/03_external_cluster_large_n100_d1_aug8_sample64_ls20/`

Existing figures were not regenerated or restyled in this cleanup.

## Final Main Results

The final retained result directory is:

```text
test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/
```

It contains per-model raw results, the LOW_FIRST LKH reference run, combined
`summary.csv`, final paper tables, and final paper figures. See
`RESULT_ARTIFACTS.md` for the exact inventory. A compact text-data copy for
paper writing is available under `paper_data/01_main_synthetic_n100_g8_d1_lkh/`.

## External Benchmark Data

The `external_benchmarks/CTSPd_SOTA/` package is present for the TSPLIB-derived
CTSP-d benchmark instances and LKH tour references. Two historical external
result sets have been restored into `paper_data/` for context:

- `02_external_cluster_large_n100_d1_aug8`: 10 `Cluster_large` n100/d1
  instances, `anchor,mds` reconstructed features, 8-fold augmentation, and
  greedy/POMO best selection.
- `03_external_cluster_large_n100_d1_aug8_sample64_ls20`: the same 10
  instances with 64 sampling runs plus 20 same-priority local-search passes.

These are external/generalization checks. The final same-distribution thesis
result remains the fixed synthetic n100/g8/d1 experiment.

## LKH Synthetic Benchmark

The repository includes a patched LKH 3.0.14 workflow for the project convention
where lower priority-group ids mean higher priority. Build it with:

```bash
cd LKH-3.0.14
make CTSPD_PRIORITY=LOW_FIRST
```

Then run the fixed synthetic benchmark with:

```bash
python scripts/run_lkh_ctspd_benchmark.py \
  --max-trials 200 \
  --runs 1 \
  --output-dir test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427/evaluations/lkh_low_first
```

The current main synthetic result including LKH is under
`test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427`.

## License

This repository contains code derived from the original POMO implementation, which is distributed under the MIT License. The original copyright notice is retained in [License.md](License.md). New CTSP-d modifications in this repository are provided under the same MIT License unless otherwise noted.

The bundled LKH source has its own upstream terms in `LKH-3.0.14/README.txt`;
review those terms before redistributing the LKH folder outside the research
context.
