import argparse
import csv
import glob
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CSTPd_bsl.POMO.CTSPd_Env import CTSPdEnv as BaselineEnv  # noqa: E402
from CSTPd_bsl.POMO.CTSPd_Model import CTSPdModel as BaselineModel  # noqa: E402
from CSTPd_cluster.CTSPd_ProblemDef import parse_ctspd_file  # noqa: E402
from CSTPd_cluster.POMO.CTSPd_Env import CTSPdEnv as ClusterEnv  # noqa: E402
from CSTPd_cluster.POMO.CTSPd_Model import CTSPdModel as ClusterModel  # noqa: E402


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
    "relation_bias_mode": "learnable",
    "relation_bias_init": 0.2,
    "relation_bias_tau": 1.0,
    "use_decoder_priority_bias": True,
    "decoder_priority_bias_mode": "learnable",
    "decoder_priority_bias_init": 0.2,
    "decoder_priority_bias_tau": 1.0,
}

INSTANCE_FIELDS = [
    "instance_id",
    "dataset_type",
    "problem_size",
    "num_groups",
    "relaxation_d",
    "model_cost",
    "cost_before_local_search",
    "local_search_improvement",
    "local_search_passes",
    "inference_time_sec",
    "augmentation_factor",
    "feature_mode",
    "decode_mode",
    "decode_run",
    "best_batch_idx",
    "best_pomo_idx",
    "is_feasible",
    "violation_count",
    "lkh_cost",
    "gap_to_lkh_percent",
    "tour_path",
    "source_path",
]

SUMMARY_FIELDS = [
    "experiment_dir",
    "checkpoint_path",
    "model_type",
    "model_variant",
    "dataset_type",
    "problem_size",
    "num_groups",
    "relaxation_d",
    "test_instance_num",
    "average_cost",
    "std_cost",
    "best_cost",
    "worst_cost",
    "feasible_rate",
    "average_violation_count",
    "total_inference_time_sec",
    "inference_time_per_instance_sec",
    "augmentation_factor",
    "feature_modes",
    "sampling_runs",
    "same_priority_ls_passes",
    "average_cost_before_local_search",
    "average_local_search_improvement",
    "average_gap_to_lkh_percent",
    "median_gap_to_lkh_percent",
    "best_gap_to_lkh_percent",
    "worst_gap_to_lkh_percent",
    "seed",
    "device",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CTSP-d checkpoints on fixed synthetic or .ctspd data.")
    parser.add_argument("--model-type", choices=["baseline", "cluster"], required=True)
    parser.add_argument(
        "--model-variant",
        default="full",
        choices=[
            "full",
            "scheduled_bias",
            "wo_group_embedding",
            "wo_fusion_gate",
            "wo_cluster_bias",
            "wo_priority_distance_bias",
            "wo_all_bias",
            "learnable_bias",
        ],
        help="Cluster ablation variant. Ignored for baseline.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint-best.pt, checkpoint-latest.pt, etc.")
    parser.add_argument("--mode", choices=["synthetic", "benchmark"], required=True)
    parser.add_argument("--dataset-file", default=None, help="Fixed synthetic .pt dataset for --mode synthetic.")
    parser.add_argument("--instance-glob", default=None, help="Glob for .ctspd files in --mode benchmark.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lkh-reference", default=None, help="Optional CSV with instance_id,lkh_cost columns.")
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=1,
        choices=[1, 8],
        help="Use 8-fold geometric augmentation and select the best POMO tour.",
    )
    parser.add_argument(
        "--benchmark-feature-modes",
        default="anchor",
        help="Comma-separated feature modes for benchmark mode: anchor,mds. Synthetic mode always uses anchor.",
    )
    parser.add_argument(
        "--sampling-runs",
        type=int,
        default=0,
        help="Extra stochastic softmax decoding runs. Greedy decoding is always included.",
    )
    parser.add_argument(
        "--same-priority-ls-passes",
        type=int,
        default=0,
        help="Maximum best-improvement swap passes among positions with the same priority. Benchmark mode only.",
    )
    parser.add_argument(
        "--save-tours",
        action="store_true",
        help="Save final benchmark tours in TSPLIB TOUR format.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(device_arg)


def set_default_device(device: torch.device) -> None:
    torch.set_default_dtype(torch.float32)
    set_default_device_fn = getattr(torch, "set_default_device", None)
    if set_default_device_fn is not None:
        set_default_device_fn(device)
    elif device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint, "epoch": None}


def cluster_params_from_state_dict(state_dict: dict[str, torch.Tensor], variant: str) -> dict[str, Any]:
    params = dict(CLUSTER_MODEL_PARAMS)
    group_key = "encoder.group_embedding.weight"
    if group_key in state_dict:
        params["num_groups"] = int(state_dict[group_key].size(0)) - 1

    if any("same_group_bias_param" in key for key in state_dict):
        runtime_values = [
            float(tensor.detach().cpu().item())
            for key, tensor in state_dict.items()
            if key.endswith("same_group_bias_runtime")
        ]
        if any(value > float(params["same_group_bias_init"]) + 1e-4 for value in runtime_values):
            params["cluster_bias_mode"] = "scheduled_residual"
        else:
            params["cluster_bias_mode"] = "learnable"
    if not any("relation_attention_bias" in key for key in state_dict):
        params["relation_bias_mode"] = "none"
    if "decoder.decoder_priority_bias_table" not in state_dict:
        params["use_decoder_priority_bias"] = False

    if variant == "wo_group_embedding":
        params["use_group_embedding"] = False
    elif variant == "wo_fusion_gate":
        params["use_group_fusion_gate"] = False
    elif variant == "wo_cluster_bias":
        params["cluster_bias_mode"] = "none"
    elif variant == "wo_priority_distance_bias":
        params["priority_distance_bias"] = 0.0
    elif variant == "wo_all_bias":
        params["cluster_bias_mode"] = "none"
        params["priority_distance_bias"] = 0.0
        params["relation_bias_mode"] = "none"
        params["use_decoder_priority_bias"] = False
    elif variant not in ("full", "scheduled_bias", "learnable_bias"):
        raise ValueError(f"Unsupported cluster variant: {variant}")

    return params


def load_model(
    model_type: str,
    model_variant: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, type, dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["model_state_dict"]
    if model_type == "baseline":
        model = BaselineModel(**BASE_MODEL_PARAMS).to(device)
        env_cls = BaselineEnv
        params = dict(BASE_MODEL_PARAMS)
    else:
        params = cluster_params_from_state_dict(state_dict, model_variant)
        model = ClusterModel(**params).to(device)
        env_cls = ClusterEnv

    model.load_state_dict(state_dict)
    model.eval()
    return model, env_cls, params, checkpoint


def read_lkh_reference(path: Optional[str]) -> dict[str, float]:
    if not path:
        return {}
    ref_path = Path(path)
    rows = csv.DictReader(ref_path.open(encoding="utf-8"))
    refs = {}
    for row in rows:
        instance_id = row.get("instance_id")
        cost = row.get("lkh_cost")
        if instance_id and cost not in (None, ""):
            refs[instance_id] = float(cost)
    return refs


def pct_gap(cost: float, reference: Optional[float]) -> Optional[float]:
    if reference is None or math.isnan(reference):
        return None
    return (cost - reference) / reference * 100.0


def calc_tour_cost(tour: list[int], dist_matrix: torch.Tensor) -> float:
    total = 0.0
    for idx, node in enumerate(tour):
        total += float(dist_matrix[node, tour[(idx + 1) % len(tour)]].item())
    return total


def calc_tour_cost_np(tour: list[int] | np.ndarray, dist_matrix: np.ndarray) -> float:
    arr = np.asarray(tour, dtype=np.int64)
    return float(dist_matrix[arr, np.roll(arr, -1)].sum())


def same_priority_swap_search(
    tour: list[int],
    dist_matrix: np.ndarray,
    priorities: np.ndarray,
    max_passes: int,
) -> tuple[list[int], float, int]:
    best = np.asarray(tour, dtype=np.int64)
    best_cost = calc_tour_cost_np(best, dist_matrix)
    pos_priority = priorities[best]
    pairs = [
        (i, j)
        for i in range(len(best) - 1)
        for j in range(i + 1, len(best))
        if pos_priority[i] == pos_priority[j]
    ]

    pass_count = 0
    for _ in range(max_passes):
        candidate: Optional[np.ndarray] = None
        candidate_cost = best_cost
        for i, j in pairs:
            swapped = best.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            cost = calc_tour_cost_np(swapped, dist_matrix)
            if cost < candidate_cost - 1e-9:
                candidate = swapped
                candidate_cost = cost
        if candidate is None:
            break
        best = candidate
        best_cost = candidate_cost
        pass_count += 1

    return best.tolist(), best_cost, pass_count


def safe_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def save_tour(tour: list[int], cost: float, instance_id: str, model_name: str, tour_dir: Path) -> Path:
    tour_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_filename_part(instance_id)}__{safe_filename_part(model_name)}__{int(round(cost))}.tour"
    path = tour_dir / filename
    lines = [
        f"NAME : {filename}",
        f"COMMENT : Length = {cost:.3f}",
        f"COMMENT : Generated by evaluate_ctspd.py for {model_name}",
        "TYPE : TOUR",
        f"DIMENSION : {len(tour)}",
        "TOUR_SECTION",
    ]
    lines.extend(str(node + 1) for node in tour)
    lines.extend(["-1", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_feature_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    if not modes:
        return ["anchor"]
    allowed = {"anchor", "mds"}
    unsupported = sorted(set(modes) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported feature modes: {unsupported}")
    return modes


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
    coords = (coords - coords_min) / np.maximum(coords_max - coords_min, 1e-8)
    return torch.from_numpy(coords.astype(np.float32))


def make_problem_features(
    base_problems: torch.Tensor,
    raw_dist: Optional[torch.Tensor],
    feature_mode: str,
) -> torch.Tensor:
    problems = base_problems.detach().cpu().clone()
    if feature_mode == "anchor":
        return problems
    if feature_mode == "mds":
        if raw_dist is None:
            raise ValueError("mds feature mode requires a raw distance matrix.")
        problems[0, :, :2] = coords_from_distance_matrix_mds(raw_dist)
        return problems
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


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


def best_tour_from_reward(selected_node_list: torch.Tensor, reward: torch.Tensor) -> tuple[list[int], float, int, int]:
    best_flat_idx = int(reward.argmax().item())
    pomo_size = int(reward.size(1))
    best_batch_idx = best_flat_idx // pomo_size
    best_pomo_idx = best_flat_idx % pomo_size
    tour = [int(node) for node in selected_node_list[best_batch_idx, best_pomo_idx].detach().cpu().tolist()]
    return tour, -float(reward[best_batch_idx, best_pomo_idx].item()), best_batch_idx, best_pomo_idx


def best_tour_by_raw_distance(selected_node_list: torch.Tensor, dist_matrix: torch.Tensor) -> tuple[list[int], float, int, int]:
    selected_cpu = selected_node_list.detach().cpu()
    dist_cpu = dist_matrix.detach().cpu()
    best_tour = None
    best_cost = None
    best_batch_idx = -1
    best_pomo_idx = -1
    for batch_idx in range(int(selected_cpu.size(0))):
        for pomo_idx in range(int(selected_cpu.size(1))):
            tour = [int(node) for node in selected_cpu[batch_idx, pomo_idx].tolist()]
            cost = calc_tour_cost(tour, dist_cpu)
            if best_cost is None or cost < best_cost:
                best_tour = tour
                best_cost = cost
                best_batch_idx = batch_idx
                best_pomo_idx = pomo_idx
    if best_tour is None or best_cost is None:
        raise RuntimeError("No tour produced.")
    return best_tour, best_cost, best_batch_idx, best_pomo_idx


def run_one_instance(
    model: torch.nn.Module,
    env_cls: type,
    problems: torch.Tensor,
    raw_dist: Optional[torch.Tensor],
    relaxation_d: int,
    num_groups: int,
    device: torch.device,
    augmentation_factor: int,
    feature_mode: str,
    decode_mode: str,
    decode_run: int,
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
            env.load_problems(
                1,
                aug_factor=augmentation_factor,
                problems=problems.to(device),
                relaxation_d=relaxation_d,
            )
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
    finally:
        if hasattr(model, "model_params"):
            model.model_params["eval_type"] = old_eval_type

    if reward is None or env.selected_node_list is None:
        raise RuntimeError("Inference did not finish correctly.")

    if raw_dist is None:
        tour, cost, best_batch_idx, best_pomo_idx = best_tour_from_reward(env.selected_node_list, reward)
    else:
        tour, cost, best_batch_idx, best_pomo_idx = best_tour_by_raw_distance(env.selected_node_list, raw_dist)
    violations, feasible = calc_feasibility(tour, problems[0, :, 2].detach().cpu(), relaxation_d)
    return {
        "tour": tour,
        "cost": cost,
        "time_sec": elapsed,
        "violation_count": violations,
        "feasible": feasible,
        "feature_mode": feature_mode,
        "decode_mode": decode_mode,
        "decode_run": decode_run,
        "best_batch_idx": best_batch_idx,
        "best_pomo_idx": best_pomo_idx,
    }


def run_best_instance(
    model: torch.nn.Module,
    env_cls: type,
    base_problems: torch.Tensor,
    raw_dist: Optional[torch.Tensor],
    relaxation_d: int,
    num_groups: int,
    device: torch.device,
    augmentation_factor: int,
    feature_modes: list[str],
    sampling_runs: int,
) -> dict[str, Any]:
    best = None
    total_time = 0.0
    tried_configs = []
    decode_plan = [("greedy", 0)]
    decode_plan.extend(("sample", run_idx + 1) for run_idx in range(sampling_runs))

    for feature_mode in feature_modes:
        problems = make_problem_features(base_problems, raw_dist, feature_mode)
        for decode_mode, decode_run in decode_plan:
            result = run_one_instance(
                model,
                env_cls,
                problems,
                raw_dist,
                relaxation_d,
                num_groups,
                device,
                augmentation_factor,
                feature_mode,
                decode_mode,
                decode_run,
            )
            total_time += float(result["time_sec"])
            tried_configs.append({
                "feature_mode": feature_mode,
                "decode_mode": decode_mode,
                "decode_run": decode_run,
                "cost": result["cost"],
                "time_sec": result["time_sec"],
            })
            if best is None or result["cost"] < best["cost"]:
                best = result

    if best is None:
        raise RuntimeError("No inference configuration was evaluated.")
    best["time_sec"] = total_time
    best["tried_configs"] = tried_configs
    return best


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_synthetic_cases(dataset_file: Path, limit: Optional[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(dataset_file, map_location="cpu")
    problems = payload["problems"]
    metadata = payload.get("metadata", {})
    if limit is not None:
        problems = problems[:limit]
    cases = []
    for idx in range(int(problems.size(0))):
        cases.append({
            "instance_id": f"synthetic_{idx:06d}",
            "dataset_type": "synthetic_fixed",
            "problems": problems[idx:idx + 1],
            "raw_dist": None,
            "problem_size": int(metadata.get("problem_size", problems.size(1))),
            "num_groups": int(metadata.get("num_groups", int(problems[idx, :, 2].max().item()))),
            "relaxation_d": int(metadata.get("relaxation_d", 1)),
            "source_path": str(dataset_file),
        })
    return cases, metadata


def load_benchmark_cases(instance_glob: str, limit: Optional[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pattern = instance_glob if Path(instance_glob).is_absolute() else str(PROJECT_ROOT / instance_glob)
    paths = sorted(Path(path).resolve() for path in glob.glob(pattern))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .ctspd instances matched: {instance_glob}")
    cases = []
    for path in paths:
        problems, raw_dist, relaxation_d, num_groups = parse_ctspd_file(str(path))
        cases.append({
            "instance_id": path.stem,
            "dataset_type": "benchmark_external",
            "problems": problems,
            "raw_dist": raw_dist,
            "problem_size": int(problems.size(1)),
            "num_groups": int(num_groups),
            "relaxation_d": int(relaxation_d),
            "source_path": str(path),
        })
    return cases, {"instance_glob": instance_glob}


def stat_or_none(values: list[float], fn) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return float(fn(clean))


def unique_or_none(values: list[Any]) -> Any:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else None


def make_output_dir(args: argparse.Namespace, dataset_type: str) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        variant = args.model_variant if args.model_type == "cluster" else "baseline"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "test_results" / f"{timestamp}_{args.model_type}_{variant}_{dataset_type}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    set_default_device(device)

    checkpoint_path = Path(args.checkpoint).resolve()
    model, env_cls, model_params, checkpoint = load_model(
        args.model_type,
        args.model_variant,
        checkpoint_path,
        device,
    )
    if args.mode == "synthetic":
        if not args.dataset_file:
            raise ValueError("--dataset-file is required for synthetic mode.")
        cases, dataset_metadata = load_synthetic_cases(Path(args.dataset_file).resolve(), args.limit)
        feature_modes = ["anchor"]
    else:
        if not args.instance_glob:
            raise ValueError("--instance-glob is required for benchmark mode.")
        cases, dataset_metadata = load_benchmark_cases(args.instance_glob, args.limit)
        feature_modes = parse_feature_modes(args.benchmark_feature_modes)

    lkh_reference = read_lkh_reference(args.lkh_reference)
    dataset_type = cases[0]["dataset_type"] if cases else args.mode
    output_dir = make_output_dir(args, dataset_type)
    model_name = args.model_variant if args.model_type == "cluster" else "baseline"
    tour_dir = output_dir / "tours"

    rows = []
    for idx, case in enumerate(cases, start=1):
        result = run_best_instance(
            model,
            env_cls,
            case["problems"],
            case["raw_dist"],
            case["relaxation_d"],
            case["num_groups"],
            device,
            args.augmentation_factor,
            feature_modes,
            args.sampling_runs,
        )
        cost_before_local_search = float(result["cost"])
        local_search_improvement = 0.0
        local_search_passes = 0
        if args.same_priority_ls_passes > 0 and case["raw_dist"] is not None:
            priorities = case["problems"][0, :, 2].detach().cpu().numpy().astype(np.int64)
            dist = case["raw_dist"].detach().cpu().numpy()
            improved_tour, improved_cost, local_search_passes = same_priority_swap_search(
                result["tour"],
                dist,
                priorities,
                args.same_priority_ls_passes,
            )
            violations, feasible = calc_feasibility(
                improved_tour,
                case["problems"][0, :, 2].detach().cpu(),
                case["relaxation_d"],
            )
            result["tour"] = improved_tour
            result["cost"] = improved_cost
            result["violation_count"] = violations
            result["feasible"] = feasible
            local_search_improvement = cost_before_local_search - float(improved_cost)

        tour_path = None
        if case["raw_dist"] is not None and (args.save_tours or args.same_priority_ls_passes > 0):
            tour_label = model_name
            if args.same_priority_ls_passes > 0:
                tour_label = f"{tour_label}_same_priority_ls{args.same_priority_ls_passes}"
            tour_path = save_tour(result["tour"], float(result["cost"]), case["instance_id"], tour_label, tour_dir)

        lkh_cost = lkh_reference.get(case["instance_id"])
        row = {
            "instance_id": case["instance_id"],
            "dataset_type": case["dataset_type"],
            "problem_size": case["problem_size"],
            "num_groups": case["num_groups"],
            "relaxation_d": case["relaxation_d"],
            "model_cost": result["cost"],
            "cost_before_local_search": cost_before_local_search,
            "local_search_improvement": local_search_improvement,
            "local_search_passes": local_search_passes,
            "inference_time_sec": result["time_sec"],
            "augmentation_factor": args.augmentation_factor,
            "feature_mode": result["feature_mode"],
            "decode_mode": result["decode_mode"],
            "decode_run": result["decode_run"],
            "best_batch_idx": result["best_batch_idx"],
            "best_pomo_idx": result["best_pomo_idx"],
            "is_feasible": result["feasible"],
            "violation_count": result["violation_count"],
            "lkh_cost": lkh_cost,
            "gap_to_lkh_percent": pct_gap(result["cost"], lkh_cost),
            "tour_path": str(tour_path) if tour_path is not None else None,
            "source_path": case["source_path"],
        }
        rows.append(row)
        print(
            "[{}/{}] {} cost={:.4f} time={:.4f}s feasible={}".format(
                idx,
                len(cases),
                case["instance_id"],
                result["cost"],
                result["time_sec"],
                result["feasible"],
            )
        )

    costs = [float(row["model_cost"]) for row in rows]
    costs_before_local_search = [float(row["cost_before_local_search"]) for row in rows]
    local_search_improvements = [float(row["local_search_improvement"]) for row in rows]
    times = [float(row["inference_time_sec"]) for row in rows]
    violations = [float(row["violation_count"]) for row in rows]
    gaps = [
        float(row["gap_to_lkh_percent"])
        for row in rows
        if row["gap_to_lkh_percent"] is not None
    ]
    total_time = sum(times)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_type": args.model_type,
        "model_variant": args.model_variant if args.model_type == "cluster" else "baseline",
        "model_params": model_params,
        "dataset_type": dataset_type,
        "dataset_metadata": dataset_metadata,
        "problem_size": unique_or_none([row["problem_size"] for row in rows]),
        "num_groups": unique_or_none([row["num_groups"] for row in rows]),
        "relaxation_d": unique_or_none([row["relaxation_d"] for row in rows]),
        "test_instance_num": len(rows),
        "average_cost": stat_or_none(costs, mean),
        "std_cost": stat_or_none(costs, pstdev),
        "best_cost": stat_or_none(costs, min),
        "worst_cost": stat_or_none(costs, max),
        "feasible_rate": mean([1.0 if row["is_feasible"] else 0.0 for row in rows]) if rows else None,
        "average_violation_count": stat_or_none(violations, mean),
        "total_inference_time_sec": total_time,
        "inference_time_per_instance_sec": total_time / len(rows) if rows else None,
        "augmentation_factor": args.augmentation_factor,
        "feature_modes": feature_modes,
        "sampling_runs": args.sampling_runs,
        "same_priority_ls_passes": args.same_priority_ls_passes,
        "average_cost_before_local_search": stat_or_none(costs_before_local_search, mean),
        "average_local_search_improvement": stat_or_none(local_search_improvements, mean),
        "average_gap_to_lkh_percent": stat_or_none(gaps, mean),
        "median_gap_to_lkh_percent": stat_or_none(gaps, median),
        "best_gap_to_lkh_percent": stat_or_none(gaps, min),
        "worst_gap_to_lkh_percent": stat_or_none(gaps, max),
        "seed": args.seed,
        "device": str(device),
        "lkh_reference": args.lkh_reference,
    }

    write_csv(output_dir / "test_instances.csv", rows, INSTANCE_FIELDS)
    write_csv(output_dir / "test_summary.csv", [{field: summary.get(field) for field in SUMMARY_FIELDS}], SUMMARY_FIELDS)
    (output_dir / "test_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {output_dir / 'test_instances.csv'}")
    print(f"Saved: {output_dir / 'test_summary.csv'}")
    print(f"Saved: {output_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
