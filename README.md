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

## Repository Layout

```text
TSP/POMO/          Original TSP POMO reference implementation
CSTPd_bsl/         CTSP-d baseline model
CSTPd_cluster/     CTSP-d cluster-aware model
CTSPd(SOTA)/       Benchmark instances, LKH tours, and original scripts
utils/             Logging, result folders, and training-curve utilities
```

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
pip install torch numpy matplotlib pytz
```

Train the cluster-aware N=20 model:

```bash
python CSTPd_cluster/POMO/train_n20.py
```

Train the baseline N=20 model:

```bash
python CSTPd_bsl/POMO/train_n20.py
```

Run single-instance inference on a benchmark `.ctspd` file:

```bash
python CSTPd_cluster/POMO/test.py --instance-file "CTSPd(SOTA)/INSTANCES/Random_small/swiss42-R-3-2-b.ctspd" --model-dir "CSTPd_cluster/POMO/result/<your_run_dir>" --checkpoint-epoch 200 --device auto
```

Run random-instance testing from a saved checkpoint:

```bash
python CSTPd_cluster/POMO/test_n20.py
```

Training and testing outputs are written under each model directory's `result/` folder. Model checkpoints and runtime logs are ignored by Git.

## Benchmark Data

`CTSPd(SOTA)/INSTANCES` contains 198 CTSP-d instances grouped into `Random_small`, `Random_large`, `Cluster_small`, and `Cluster_large`. `CTSPd(SOTA)/TOURS` contains LKH solution tours used for best-known-length comparison.

For file-based testing, the model input uses reconstructed normalized 2D features from the benchmark distance matrix, while the final reported tour length is computed using the original distance matrix.

## License

This repository contains code derived from the original POMO implementation, which is distributed under the MIT License. The original copyright notice is retained in [License.md](License.md). New CTSP-d modifications in this repository are provided under the same MIT License unless otherwise noted.
