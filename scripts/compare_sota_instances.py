import argparse
import csv
import glob
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional, Type

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CSTPd_bsl.POMO.CTSPd_Env import CTSPdEnv as BslEnv  # noqa: E402
from CSTPd_bsl.POMO.CTSPd_Model import CTSPdModel as BslModel  # noqa: E402
from CSTPd_cluster.CTSPd_ProblemDef import parse_ctspd_file  # noqa: E402
from CSTPd_cluster.POMO.CTSPd_Env import CTSPdEnv as ClusterEnv  # noqa: E402
from CSTPd_cluster.POMO.CTSPd_Model import CTSPdModel as ClusterModel  # noqa: E402


DEFAULT_BSL_CHECKPOINT = (
    PROJECT_ROOT / "CSTPd_bsl" / "POMO" / "result"
    / "21日_13点43分_baseline_n100_d1" / "checkpoint-best.pt"
)
DEFAULT_CLUSTER_CHECKPOINT = (
    PROJECT_ROOT / "CSTPd_cluster" / "POMO" / "result"
    / "21日_12点17分_cluster_n100_d1_resume_e116_to160" / "checkpoint-best.pt"
)
DEFAULT_INSTANCE_GLOB = (
    PROJECT_ROOT / "CTSPd(SOTA)" / "INSTANCES" / "Cluster_large" / "*100-C-*-1-*.ctspd"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "comparison_results"


BASE_MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


CLUSTER_MODEL_PARAMS = {
    **BASE_MODEL_PARAMS,
    "num_groups": 8,
    "use_group_embedding": True,
    "use_group_fusion_gate": True,
    "cluster_bias_mode": "scheduled",
    "same_group_bias_init": 0.1,
    "same_group_bias_final": 1.25,
    "same_group_bias_warmup_epochs": 20,
    "priority_distance_bias": 0.15,
    "priority_distance_tau": 1.0,
    # Keep old-cluster checkpoints loadable after enhanced cluster features were added.
    "relation_bias_mode": "none",
    "use_decoder_priority_bias": False,
}


CSV_FIELDS = [
    "instance",
    "category",
    "n_nodes",
    "num_groups",
    "relaxation_d",
    "lkh_cost",
    "bsl_cost",
    "cluster_cost",
    "bsl_gap_to_lkh_percent",
    "cluster_gap_to_lkh_percent",
    "cluster_minus_bsl",
    "cluster_improvement_vs_bsl_percent",
    "winner",
    "bsl_time_sec",
    "cluster_time_sec",
    "bsl_total_search_time_sec",
    "cluster_total_search_time_sec",
    "bsl_best_batch",
    "bsl_best_pomo",
    "cluster_best_batch",
    "cluster_best_pomo",
    "bsl_feature_mode",
    "cluster_feature_mode",
    "bsl_decode_mode",
    "cluster_decode_mode",
    "bsl_decode_run",
    "cluster_decode_run",
    "bsl_violation_count",
    "cluster_violation_count",
    "bsl_feasible",
    "cluster_feasible",
    "lkh_tour_file",
    "bsl_tour_file",
    "cluster_tour_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CTSP-d baseline and cluster models on SOTA .ctspd instances."
    )
    parser.add_argument(
        "--instance-glob",
        default=str(DEFAULT_INSTANCE_GLOB),
        help="Glob for .ctspd instances. Default selects Cluster_large n100 d1 instances.",
    )
    parser.add_argument(
        "--bsl-checkpoint",
        default=str(DEFAULT_BSL_CHECKPOINT),
        help="Baseline checkpoint path.",
    )
    parser.add_argument(
        "--cluster-checkpoint",
        default=str(DEFAULT_CLUSTER_CHECKPOINT),
        help="Cluster checkpoint path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for comparison subfolders.",
    )
    parser.add_argument(
        "--comparison-name",
        default="sota_cluster_large_n100_d1__bsl_vs_cluster__custom_inference",
        help="Readable name for the comparison subfolder.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of instances.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable 8-fold augmentation on reconstructed 2D features. Off by default for real instances.",
    )
    parser.add_argument(
        "--feature-modes",
        default="anchor",
        help="Comma-separated coordinate feature modes to try: anchor,mds. Example: anchor,mds",
    )
    parser.add_argument(
        "--sampling-runs",
        type=int,
        default=0,
        help="Extra stochastic softmax decoding runs per model/config. Greedy is always included.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for stochastic decoding.",
    )
    parser.add_argument(
        "--no-save-tours",
        action="store_true",
        help="Do not save generated best tours.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def set_default_device(device: torch.device) -> None:
    torch.set_default_dtype(torch.float32)
    set_default_device_fn = getattr(torch, "set_default_device", None)
    if set_default_device_fn is not None:
        set_default_device_fn(device)
    elif device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def resolve_instances(instance_glob: str, limit: Optional[int]) -> list[Path]:
    pattern = instance_glob
    if not Path(pattern).is_absolute():
        pattern = str(PROJECT_ROOT / pattern)
    instances = sorted(Path(path) for path in glob.glob(pattern))
    if limit is not None:
        instances = instances[:limit]
    if not instances:
        raise FileNotFoundError(f"No instances matched: {instance_glob}")
    return [path.resolve() for path in instances]


def parse_feature_modes(raw_modes: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw_modes.split(",") if mode.strip()]
    if not modes:
        raise ValueError("--feature-modes must include at least one mode.")
    supported = {"anchor", "mds"}
    unknown = [mode for mode in modes if mode not in supported]
    if unknown:
        raise ValueError(f"Unsupported feature modes: {unknown}. Supported: {sorted(supported)}")
    return list(dict.fromkeys(modes))


def make_output_dir(output_root: Path, comparison_name: str, instance_count: int) -> Path:
    base = output_root / f"{comparison_name}__{instance_count}_instances"
    if not base.exists():
        base.mkdir(parents=True)
        return base

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{base.name}__run{timestamp}"
    output_dir.mkdir(parents=True)
    return output_dir


def infer_cluster_num_groups(state_dict: dict[str, torch.Tensor]) -> Optional[int]:
    key = "encoder.group_embedding.weight"
    if key not in state_dict:
        return None
    return int(state_dict[key].size(0)) - 1


def build_cluster_params(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    params = dict(CLUSTER_MODEL_PARAMS)
    num_groups = infer_cluster_num_groups(state_dict)
    if num_groups is not None:
        params["num_groups"] = num_groups
    if any("relation_attention_bias" in key for key in state_dict):
        params["relation_bias_mode"] = "learnable"
    if "decoder.decoder_priority_bias_table" in state_dict:
        params["use_decoder_priority_bias"] = True
        params["decoder_priority_bias_mode"] = "learnable"
    return params


def load_model(
    model_cls: Type[torch.nn.Module],
    checkpoint_path: Path,
    model_params: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    if model_cls is ClusterModel:
        model_params = build_cluster_params(state_dict)
    model = model_cls(**model_params).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {
        "path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "model_params": model_params,
    }


def category_from_instance(instance_path: Path) -> str:
    try:
        rel = instance_path.relative_to(PROJECT_ROOT / "CTSPd(SOTA)" / "INSTANCES")
        return rel.parts[0]
    except ValueError:
        return instance_path.parent.name


def find_lkh_tour(instance_path: Path) -> tuple[Optional[float], Optional[Path]]:
    category = category_from_instance(instance_path)
    tour_dir = PROJECT_ROOT / "CTSPd(SOTA)" / "TOURS" / category
    stem = instance_path.stem
    filename_pattern = re.compile(rf"^{re.escape(stem)}\.(\d+(?:\.\d+)?)\.tour$")

    best_cost: Optional[float] = None
    best_file: Optional[Path] = None
    candidates = list(tour_dir.glob(f"{stem}.*.tour"))
    if not candidates:
        candidates = list((PROJECT_ROOT / "CTSPd(SOTA)" / "TOURS").rglob(f"{stem}.*.tour"))

    for tour_file in candidates:
        cost = None
        match = filename_pattern.match(tour_file.name)
        if match is not None:
            cost = float(match.group(1))
        else:
            text = tour_file.read_text(encoding="utf-8", errors="ignore")
            comment_match = re.search(r"Length\s*=\s*(\d+(?:\.\d+)?)", text)
            if comment_match is not None:
                cost = float(comment_match.group(1))
        if cost is not None and (best_cost is None or cost < best_cost):
            best_cost = cost
            best_file = tour_file

    return best_cost, best_file


def calc_tour_cost(tour: list[int], dist_matrix: torch.Tensor) -> float:
    total = 0.0
    n_nodes = len(tour)
    for idx, node in enumerate(tour):
        next_node = tour[(idx + 1) % n_nodes]
        total += float(dist_matrix[node, next_node].item())
    return total


def coords_from_distance_matrix_mds(dist_matrix: torch.Tensor) -> torch.Tensor:
    dist = dist_matrix.detach().cpu().numpy().astype(np.float64)
    n_nodes = dist.shape[0]
    if n_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    squared = dist ** 2
    centering = np.eye(n_nodes) - np.ones((n_nodes, n_nodes)) / n_nodes
    gram = -0.5 * centering @ squared @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    top = order[:2]
    vals = np.maximum(eigvals[top], 0.0)
    coords = eigvecs[:, top] * np.sqrt(vals)[None, :]

    if coords.shape[1] < 2 or np.allclose(coords, 0.0):
        coords = np.stack([dist.mean(axis=1), dist.max(axis=1)], axis=1)

    coords_min = coords.min(axis=0, keepdims=True)
    coords_max = coords.max(axis=0, keepdims=True)
    denom = np.maximum(coords_max - coords_min, 1e-8)
    coords = (coords - coords_min) / denom
    return torch.from_numpy(coords.astype(np.float32))


def make_problem_features(
    base_problems: torch.Tensor,
    raw_dist: torch.Tensor,
    feature_mode: str,
) -> torch.Tensor:
    problems = base_problems.detach().cpu().clone()
    if feature_mode == "anchor":
        return problems
    if feature_mode == "mds":
        problems[0, :, :2] = coords_from_distance_matrix_mds(raw_dist)
        return problems
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def calc_feasibility(tour: list[int], priorities: torch.Tensor, relaxation_d: int) -> tuple[int, bool]:
    n_nodes = int(priorities.numel())
    duplicate_or_missing = len(set(tour)) != n_nodes or len(tour) != n_nodes
    unvisited = set(range(n_nodes))
    violations = 0

    for node in tour:
        if node not in unvisited:
            violations += 1
            continue
        current_min = min(float(priorities[idx].item()) for idx in unvisited)
        node_priority = float(priorities[node].item())
        if node_priority < current_min or node_priority > current_min + relaxation_d:
            violations += 1
        unvisited.remove(node)

    if duplicate_or_missing or unvisited:
        violations += len(unvisited)
    return violations, violations == 0


def best_raw_tour(selected_node_list: torch.Tensor, dist_matrix: torch.Tensor) -> tuple[list[int], float, int, int]:
    selected_cpu = selected_node_list.detach().cpu()
    dist_cpu = dist_matrix.detach().cpu()

    best_tour: Optional[list[int]] = None
    best_cost: Optional[float] = None
    best_batch = -1
    best_pomo = -1

    for batch_idx in range(int(selected_cpu.size(0))):
        for pomo_idx in range(int(selected_cpu.size(1))):
            tour = [int(node) for node in selected_cpu[batch_idx, pomo_idx].tolist()]
            cost = calc_tour_cost(tour, dist_cpu)
            if best_cost is None or cost < best_cost:
                best_tour = tour
                best_cost = cost
                best_batch = batch_idx
                best_pomo = pomo_idx

    if best_tour is None or best_cost is None:
        raise RuntimeError("No POMO tour was produced.")
    return best_tour, best_cost, best_batch, best_pomo


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_model_on_instance(
    model: torch.nn.Module,
    env_cls: Type[Any],
    problems: torch.Tensor,
    raw_dist: torch.Tensor,
    relaxation_d: int,
    num_groups: int,
    device: torch.device,
    aug_factor: int,
    decode_mode: str,
    decode_run: int,
    feature_mode: str,
) -> dict[str, Any]:
    n_nodes = int(problems.size(1))
    env = env_cls(
        problem_size=n_nodes,
        pomo_size=n_nodes,
        num_groups=num_groups,
        relaxation_d=relaxation_d,
    )

    old_eval_type = getattr(model, "model_params", {}).get("eval_type")
    if hasattr(model, "model_params"):
        model.model_params["eval_type"] = "softmax" if decode_mode == "sample" else "argmax"
    try:
        synchronize_if_needed(device)
        start = time.perf_counter()
        with torch.inference_mode():
            env.load_problems(1, aug_factor=aug_factor, problems=problems.to(device), relaxation_d=relaxation_d)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, _, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, _, done = env.step(selected)

        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
    finally:
        if hasattr(model, "model_params"):
            model.model_params["eval_type"] = old_eval_type

    if env.selected_node_list is None:
        raise RuntimeError("Inference did not produce selected_node_list.")

    best_tour, best_cost, best_batch, best_pomo = best_raw_tour(env.selected_node_list, raw_dist)
    violations, feasible = calc_feasibility(best_tour, problems[0, :, 2].detach().cpu(), relaxation_d)
    return {
        "tour": best_tour,
        "cost": best_cost,
        "best_batch": best_batch,
        "best_pomo": best_pomo,
        "time_sec": elapsed,
        "violation_count": violations,
        "feasible": feasible,
        "feature_mode": feature_mode,
        "decode_mode": decode_mode,
        "decode_run": decode_run,
    }


def run_best_model_on_instance(
    model: torch.nn.Module,
    env_cls: Type[Any],
    base_problems: torch.Tensor,
    raw_dist: torch.Tensor,
    relaxation_d: int,
    num_groups: int,
    device: torch.device,
    aug_factor: int,
    feature_modes: list[str],
    sampling_runs: int,
) -> dict[str, Any]:
    best: Optional[dict[str, Any]] = None
    total_time = 0.0
    tried_configs = []

    for feature_mode in feature_modes:
        problems = make_problem_features(base_problems, raw_dist, feature_mode)
        decode_plan = [("greedy", 0)]
        decode_plan.extend(("sample", run_idx + 1) for run_idx in range(sampling_runs))

        for decode_mode, decode_run in decode_plan:
            result = run_model_on_instance(
                model,
                env_cls,
                problems,
                raw_dist,
                relaxation_d,
                num_groups,
                device,
                aug_factor,
                decode_mode,
                decode_run,
                feature_mode,
            )
            total_time += float(result["time_sec"])
            tried_configs.append(
                {
                    "feature_mode": feature_mode,
                    "decode_mode": decode_mode,
                    "decode_run": decode_run,
                    "cost": result["cost"],
                    "time_sec": result["time_sec"],
                }
            )
            if best is None or result["cost"] < best["cost"]:
                best = result

    if best is None:
        raise RuntimeError("No inference configuration was evaluated.")
    best["total_time_sec"] = total_time
    best["tried_configs"] = tried_configs
    return best


def save_tour(tour: list[int], cost: float, instance_path: Path, model_name: str, tour_dir: Path) -> Path:
    tour_dir.mkdir(parents=True, exist_ok=True)
    rounded_cost = int(round(cost))
    filename = f"{instance_path.stem}__{model_name}__{rounded_cost}.tour"
    path = tour_dir / filename
    lines = [
        f"NAME : {filename}",
        f"COMMENT : Length = {cost:.3f}",
        f"COMMENT : Generated by compare_sota_instances.py for {model_name}",
        "TYPE : TOUR",
        f"DIMENSION : {len(tour)}",
        "TOUR_SECTION",
    ]
    lines.extend(str(node + 1) for node in tour)
    lines.extend(["-1", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def pct_gap(cost: Optional[float], reference: Optional[float]) -> Optional[float]:
    if cost is None or reference is None:
        return None
    return (cost - reference) / reference * 100.0


def pct_improvement(old_cost: float, new_cost: float) -> float:
    return (old_cost - new_cost) / old_cost * 100.0


def winner_name(bsl_cost: float, cluster_cost: float) -> str:
    if cluster_cost < bsl_cost:
        return "cluster"
    if bsl_cost < cluster_cost:
        return "bsl"
    return "tie"


def numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(key: str) -> dict[str, Optional[float]]:
        values = numeric_values(rows, key)
        if not values:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }

    winners = [row["winner"] for row in rows]
    return {
        "instance_count": len(rows),
        "bsl_cost": stats("bsl_cost"),
        "cluster_cost": stats("cluster_cost"),
        "lkh_cost": stats("lkh_cost"),
        "bsl_gap_to_lkh_percent": stats("bsl_gap_to_lkh_percent"),
        "cluster_gap_to_lkh_percent": stats("cluster_gap_to_lkh_percent"),
        "cluster_minus_bsl": stats("cluster_minus_bsl"),
        "cluster_improvement_vs_bsl_percent": stats("cluster_improvement_vs_bsl_percent"),
        "bsl_time_sec": stats("bsl_time_sec"),
        "cluster_time_sec": stats("cluster_time_sec"),
        "bsl_total_search_time_sec": stats("bsl_total_search_time_sec"),
        "cluster_total_search_time_sec": stats("cluster_total_search_time_sec"),
        "cluster_win_count": winners.count("cluster"),
        "bsl_win_count": winners.count("bsl"),
        "tie_count": winners.count("tie"),
        "cluster_win_rate": winners.count("cluster") / len(rows) if rows else None,
        "all_bsl_feasible": all(bool(row["bsl_feasible"]) for row in rows),
        "all_cluster_feasible": all(bool(row["cluster_feasible"]) for row in rows),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_default_device(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    instances = resolve_instances(args.instance_glob, args.limit)
    feature_modes = parse_feature_modes(args.feature_modes)
    aug_factor = 8 if args.augmentation else 1
    output_dir = make_output_dir(Path(args.output_root), args.comparison_name, len(instances))
    tour_dir = output_dir / "tours"

    print(f"Device: {device}")
    print(f"Instances: {len(instances)}")
    print(f"Augmentation: {aug_factor}")
    print(f"Feature modes: {feature_modes}")
    print(f"Sampling runs per feature/model: {args.sampling_runs}")
    print(f"Output: {output_dir}")
    if args.augmentation:
        print("Warning: augmentation uses reconstructed features, not the original distance-matrix geometry.")

    bsl_model, bsl_info = load_model(BslModel, Path(args.bsl_checkpoint), dict(BASE_MODEL_PARAMS), device)
    cluster_model, cluster_info = load_model(
        ClusterModel,
        Path(args.cluster_checkpoint),
        dict(CLUSTER_MODEL_PARAMS),
        device,
    )

    rows: list[dict[str, Any]] = []
    for instance_path in instances:
        problems, raw_dist, relaxation_d, num_groups = parse_ctspd_file(str(instance_path))
        n_nodes = int(problems.size(1))
        category = category_from_instance(instance_path)
        lkh_cost, lkh_tour_file = find_lkh_tour(instance_path)

        print(f"[{len(rows) + 1}/{len(instances)}] {instance_path.name}: n={n_nodes}, groups={num_groups}, d={relaxation_d}")

        bsl_result = run_best_model_on_instance(
            bsl_model,
            BslEnv,
            problems,
            raw_dist,
            relaxation_d,
            num_groups,
            device,
            aug_factor,
            feature_modes,
            args.sampling_runs,
        )
        cluster_result = run_best_model_on_instance(
            cluster_model,
            ClusterEnv,
            problems,
            raw_dist,
            relaxation_d,
            num_groups,
            device,
            aug_factor,
            feature_modes,
            args.sampling_runs,
        )

        bsl_tour_file = None
        cluster_tour_file = None
        if not args.no_save_tours:
            bsl_tour_file = save_tour(bsl_result["tour"], bsl_result["cost"], instance_path, "bsl", tour_dir)
            cluster_tour_file = save_tour(
                cluster_result["tour"],
                cluster_result["cost"],
                instance_path,
                "cluster",
                tour_dir,
            )

        row = {
            "instance": instance_path.name,
            "category": category,
            "n_nodes": n_nodes,
            "num_groups": num_groups,
            "relaxation_d": relaxation_d,
            "lkh_cost": lkh_cost,
            "bsl_cost": bsl_result["cost"],
            "cluster_cost": cluster_result["cost"],
            "bsl_gap_to_lkh_percent": pct_gap(bsl_result["cost"], lkh_cost),
            "cluster_gap_to_lkh_percent": pct_gap(cluster_result["cost"], lkh_cost),
            "cluster_minus_bsl": cluster_result["cost"] - bsl_result["cost"],
            "cluster_improvement_vs_bsl_percent": pct_improvement(bsl_result["cost"], cluster_result["cost"]),
            "winner": winner_name(bsl_result["cost"], cluster_result["cost"]),
            "bsl_time_sec": bsl_result["time_sec"],
            "cluster_time_sec": cluster_result["time_sec"],
            "bsl_total_search_time_sec": bsl_result["total_time_sec"],
            "cluster_total_search_time_sec": cluster_result["total_time_sec"],
            "bsl_best_batch": bsl_result["best_batch"],
            "bsl_best_pomo": bsl_result["best_pomo"],
            "cluster_best_batch": cluster_result["best_batch"],
            "cluster_best_pomo": cluster_result["best_pomo"],
            "bsl_feature_mode": bsl_result["feature_mode"],
            "cluster_feature_mode": cluster_result["feature_mode"],
            "bsl_decode_mode": bsl_result["decode_mode"],
            "cluster_decode_mode": cluster_result["decode_mode"],
            "bsl_decode_run": bsl_result["decode_run"],
            "cluster_decode_run": cluster_result["decode_run"],
            "bsl_violation_count": bsl_result["violation_count"],
            "cluster_violation_count": cluster_result["violation_count"],
            "bsl_feasible": bsl_result["feasible"],
            "cluster_feasible": cluster_result["feasible"],
            "lkh_tour_file": str(lkh_tour_file) if lkh_tour_file else None,
            "bsl_tour_file": str(bsl_tour_file) if bsl_tour_file else None,
            "cluster_tour_file": str(cluster_tour_file) if cluster_tour_file else None,
        }
        rows.append(row)
        print(
            "    bsl={:.0f}({}/{}), cluster={:.0f}({}/{}), lkh={}, winner={}, cluster_vs_bsl={:.3f}%".format(
                row["bsl_cost"],
                row["bsl_feature_mode"],
                row["bsl_decode_mode"],
                row["cluster_cost"],
                row["cluster_feature_mode"],
                row["cluster_decode_mode"],
                "{:.0f}".format(lkh_cost) if lkh_cost is not None else "NA",
                row["winner"],
                row["cluster_improvement_vs_bsl_percent"],
            )
        )

    csv_path = output_dir / "instances.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "augmentation_factor": aug_factor,
        "feature_modes": feature_modes,
        "sampling_runs": args.sampling_runs,
        "seed": args.seed,
        "instance_glob": args.instance_glob,
        "selected_instances": [str(path) for path in instances],
        "bsl_checkpoint": bsl_info,
        "cluster_checkpoint": cluster_info,
        "summary": summarize(rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
