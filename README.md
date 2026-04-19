# POMO for CTSP-d

本仓库基于 POMO 的 PyTorch 实现，扩展到 **Clustered Traveling Salesman Problem with d-relaxed priority rule (CTSP-d)**。代码保留了原始 TSP/POMO 作为参考，同时新增了 CTSP-d 的 baseline 版本和 cluster-aware 版本。

CTSP-d 中每个节点包含 `(x, y, priority)` 三个特征。模型需要构造一个 Hamiltonian tour，并满足 d-relaxed priority rule：在任一步，若未访问节点中的最高优先级为 `p`（代码中优先级数值越小越高），则下一步只能选择优先级位于 `[p, p + d]` 的未访问节点。

## 目录结构

```text
.
├── TSP/POMO/                 # 原始 TSP POMO 训练、测试、模型和环境
├── CSTPd_bsl/                # CTSP-d baseline：直接把 priority 作为第 3 维输入
│   ├── CTSPd_ProblemDef.py   # 随机问题生成、8-fold 增强、.ctspd 解析
│   └── POMO/
├── CSTPd_cluster/            # CTSP-d cluster-aware 版本
│   ├── CTSPd_ProblemDef.py
│   └── POMO/
├── CTSPd(SOTA)/              # CTSP-d benchmark instances、LKH tours 和原始脚本
└── utils/                    # 日志、结果目录、训练曲线等工具
```

## 当前实现

- **CTSP-d 环境**：`CTSPd_Env.py` 在 POMO rollout 中维护已访问 mask、当前未访问节点的最高优先级，以及 d-relaxed priority mask。最终 reward 为 tour length 的负值。
- **随机训练数据**：`get_random_problems()` 生成 `[x, y, priority]`，并尽量保证每个 priority group 至少出现一次。
- **8-fold 几何增强**：对随机二维坐标做翻转/交换增强，同时保持 priority 不变。
- **benchmark 文件解析**：`parse_ctspd_file()` 支持读取 `.ctspd` 中的 `DIMENSION`、`GROUPS`、`RELAXATION_LEVEL`、`EDGE_WEIGHT_SECTION` 和 `GROUP_SECTION`。
- **真实距离评估**：文件测试时会用 `.ctspd` 的原始距离矩阵计算真实 tour length；模型输入使用由距离矩阵重建并归一化后的二维特征。
- **tour 保存**：单文件测试脚本会把模型生成的路径保存为 `.tour`，并尝试从 `CTSPd(SOTA)/TOURS` 中读取 best-known length 计算 gap。

## 两个 CTSP-d 版本

### Baseline

路径：`CSTPd_bsl/POMO/`

Baseline 模型把节点特征 `[x, y, priority]` 直接输入线性 embedding，仍使用原始 POMO encoder/decoder 结构。环境侧已经加入 CTSP-d 的优先级约束 mask。

常用入口：

```bash
python CSTPd_bsl/POMO/train_n20.py
python CSTPd_bsl/POMO/train_n50.py
python CSTPd_bsl/POMO/train_n100.py
python CSTPd_bsl/POMO/test.py
```

### Cluster-aware

路径：`CSTPd_cluster/POMO/`

Cluster-aware 模型对坐标和 priority group 分开建模：

- `coord_embedding`: 对 `(x, y)` 做坐标 embedding；
- `group_embedding`: 为每个优先级组学习可训练 embedding；
- `fusion`: 融合坐标特征和组特征；
- encoder self-attention 中加入同组 attention bias，使模型显式感知 priority group。

常用入口：

```bash
python CSTPd_cluster/POMO/train_n20.py
python CSTPd_cluster/POMO/train_n50.py
python CSTPd_cluster/POMO/train_n100.py
python CSTPd_cluster/POMO/test.py
```

`train_n20.py` 中已经包含 `num_groups` 一致性检查。cluster-aware 模型加载 checkpoint 时也会根据 `encoder.group_embedding.weight` 推断 checkpoint 支持的组数。

## 环境依赖

代码使用了 `torch.set_default_device()` 和 Python 现代类型标注，建议环境：

```bash
conda create -n py310-env python=3.10
conda activate py310-env
pip install torch numpy matplotlib pytz
```

如果没有 CUDA，可以在训练/测试脚本中把 `DEBUG_MODE = True` 或手动设置 `USE_CUDA = False`。

## 训练

以 cluster-aware N=20 为例：

```bash
python CSTPd_cluster/POMO/train_n20.py
```

主要参数在脚本顶部配置：

- `env_params`: `problem_size`、`pomo_size`、`num_groups`、`relaxation_d`
- `model_params`: embedding 维度、encoder 层数、head 数、`num_groups`
- `trainer_params`: epoch 数、episode 数、batch size、checkpoint 保存间隔
- `optimizer_params`: Adam 与 scheduler 参数

训练输出会写入对应目录下的 `result/YYYYMMDD_HHMMSS_<desc>/`，包含日志、checkpoint 和训练曲线。`result/` 目录及模型文件已在 `.gitignore` 中忽略，避免把大模型文件提交到 Git。

## 测试随机实例

`test_n20.py` / `test_n100.py` 使用脚本内的 `tester_params` 测试随机生成的 CTSP-d 实例：

```bash
python CSTPd_cluster/POMO/test_n20.py
```

如果配置的 checkpoint 路径不存在，`CTSPd_Tester.py` 会在对应 `result/` 目录下搜索匹配 epoch 的 checkpoint。

## 测试 benchmark .ctspd 文件

单文件测试脚本支持命令行参数，默认实例为：

```text
CTSPd(SOTA)/INSTANCES/Random_small/swiss42-R-3-2-b.ctspd
```

示例：

```bash
python CSTPd_cluster/POMO/test.py --instance-file "CTSPd(SOTA)/INSTANCES/Random_small/swiss42-R-3-2-b.ctspd" --model-dir "CSTPd_cluster/POMO/result/<your_run_dir>" --checkpoint-epoch 200 --device auto
```

Baseline 版本使用：

```bash
python CSTPd_bsl/POMO/test.py --device auto
```

测试输出包括：

- 模型在归一化二维特征上的 objective；
- 使用原始距离矩阵计算的 real tour length；
- best known length 和 gap（若 `CTSPd(SOTA)/TOURS` 中存在对应 `.tour`）；
- 生成的 `.tour` 文件路径。

注意：benchmark `.ctspd` 文件只提供距离矩阵，二维坐标是由距离矩阵重建的模型特征，因此文件测试中的真实长度始终以原始距离矩阵为准。批量文件测试逻辑中会关闭几何增强，避免对重建坐标做不保真的增强。

## CTSPd(SOTA) 数据

`CTSPd(SOTA)/INSTANCES` 当前包含 198 个 `.ctspd` 实例，分为：

- `Random_small`
- `Random_large`
- `Cluster_small`
- `Cluster_large`

`CTSPd(SOTA)/TOURS` 包含对应的 LKH solution tours，用于读取 best-known length 和对比 gap。

## 与原始 POMO 的关系

本项目保留了原始 POMO 的 TSP 代码在 `TSP/POMO/` 下，CTSP-d 相关改动集中在 `CSTPd_bsl/` 和 `CSTPd_cluster/`。原始 POMO README 中提到的 CVRP 目录在当前仓库中并不是主要维护对象；当前仓库的重点是 CTSP-d。

原始 POMO 代码遵循 MIT License，详见 `License.md`。
