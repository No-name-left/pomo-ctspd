# Comparison Results

This folder stores CTSP-d model comparison outputs. Each subfolder is one completed
experiment and should contain:

```text
summary.json    aggregate metrics
instances.csv   per-instance costs, gaps, winning model, selected decode setup
tours/          generated best tours, when tour saving is enabled
```

Only the enhanced SOTA benchmark results are kept here. Older no-augmentation
greedy results and temporary smoke-test outputs were removed to avoid confusing
them with the recommended benchmark pipeline.

## Current Results

```text
sota_cluster_large_n100_d1__bsl_vs_cluster__enhanced_inference_mds_aug_sample64__10_instances
```

Recommended neural-only enhanced inference result. This uses MDS feature
reconstruction, 8-fold augmentation, greedy decoding plus 64 stochastic sampling
runs, and then selects the best generated tour by the original CTSP-d distance
matrix.

```text
sota_cluster_large_n100_d1__bsl_vs_cluster__enhanced_inference_mds_aug_sample64_plus_same_priority_ls20__10_instances
```

Enhanced inference plus same-priority local search. This starts from the first
result and applies a feasible local search that swaps nodes occupying positions
with the same priority. Because the priority sequence is unchanged, the
d-relaxed priority rule remains feasible.

## Recommended Commands

Run enhanced inference without local search:

```bash
python scripts/compare_sota_instances.py \
  --comparison-name sota_cluster_large_n100_d1__bsl_vs_cluster__enhanced_inference_mds_aug_sample64 \
  --feature-modes anchor,mds \
  --augmentation \
  --sampling-runs 64 \
  --seed 20260421
```

Then run same-priority local search on the enhanced inference result:

```bash
python scripts/postprocess_same_priority_local_search.py --passes 20
```

The postprocess script defaults to:

```text
comparison_results/sota_cluster_large_n100_d1__bsl_vs_cluster__enhanced_inference_mds_aug_sample64__10_instances
```

Use `--source-dir` if you want to postprocess another comparison folder.

## Instance Reading

SOTA instances live under:

```text
CTSPd(SOTA)/INSTANCES/
```

The current recommended command evaluates:

```text
CTSPd(SOTA)/INSTANCES/Cluster_large/*100-C-*-1-*.ctspd
```

This selects the 10 `Cluster_large`, `n=100`, `d=1` instances that match the
currently trained n100/d1 checkpoints.

The script reads:

```text
DIMENSION
GROUPS
RELAXATION_LEVEL
EDGE_WEIGHT_SECTION
GROUP_SECTION
```

The model input is `[x, y, priority]`, but the SOTA CTSP-d files use explicit
distance matrices instead of original coordinates. Therefore, the script builds
2D features from the distance matrix for model inference, while final costs are
always computed from the original distance matrix.

## LKH/SOTA Reference

LKH reference costs are read from:

```text
CTSPd(SOTA)/TOURS/
```

The PDFs under `CTSPd(SOTA)/RESULTS/` are useful for human inspection, but the
automated comparison uses `.tour` files because they are machine-readable and
contain the LKH tour length. The comparison script does not use the LKH tour to
construct or improve model tours; it only uses the LKH cost to compute gaps.

## Enhanced Inference Details

The enhanced inference pipeline does not retrain or modify the model weights.
It only improves how candidate tours are generated and selected.

`--feature-modes anchor,mds`

Tries two ways to convert the distance matrix into 2D model features:

```text
anchor    older fallback using distances to two anchor nodes
mds       recommended classical MDS projection from the distance matrix
```

In the current SOTA n100/d1 results, the winning candidates all came from `mds`.

`--augmentation`

Runs 8-fold coordinate augmentation on the reconstructed 2D features. This gives
the model several equivalent coordinate orientations. Since the final score is
computed from the original distance matrix, augmentation only changes candidate
generation, not the evaluation metric.

`--sampling-runs 64`

For each feature mode, the script always runs one greedy decode and then 64
stochastic softmax decodes. All produced POMO tours are evaluated by the original
distance matrix, and the shortest feasible candidate is selected.

`--seed 20260421`

Fixes stochastic sampling for reproducibility.

## Local Search Details

`postprocess_same_priority_local_search.py` performs same-priority swap local
search. It only swaps two nodes if their priorities at the current tour positions
are equal. This keeps the tour's priority sequence unchanged, so feasibility is
preserved when the source tour is feasible.

This stage should be reported separately from pure neural inference:

```text
Enhanced inference
Enhanced inference + same-priority local search
```

## Reporting Notes

Use `summary.json` for aggregate metrics:

```text
bsl_cost
cluster_cost
bsl_gap_to_lkh_percent
cluster_gap_to_lkh_percent
cluster_improvement_vs_bsl_percent
cluster_win_count
all_bsl_feasible
all_cluster_feasible
```

Use `instances.csv` for per-instance analysis and for checking which feature
mode / decode mode produced the selected tour:

```text
bsl_feature_mode
cluster_feature_mode
bsl_decode_mode
cluster_decode_mode
bsl_decode_run
cluster_decode_run
```

For clean thesis tables, keep these rows separate:

```text
1. Enhanced inference: MDS + augmentation + sampling
2. Enhanced inference + same-priority local search
```

Do not mix local-search results into the neural-only row.
