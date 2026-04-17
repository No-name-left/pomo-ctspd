# pylint: disable=wrong-import-position
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CSTPd_cluster.POMO.CTSPd_Env import CTSPdEnv  # noqa: E402
from CSTPd_cluster.POMO.CTSPd_Model import CTSPdModel  # noqa: E402
from CSTPd_cluster.CTSPd_ProblemDef import parse_ctspd_file  # noqa: E402


DEFAULT_MODEL_PARAMS = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


def project_root():
    return PROJECT_ROOT


def result_root():
    return Path(__file__).resolve().parent / "result"


def default_instance_file():
    return project_root() / "CTSPd(SOTA)" / "INSTANCES" / "Random_small" / "swiss42-R-3-2-b.ctspd"


def default_model_dir():
    preferred = result_root() / "train__ctspd_n20_cluster"
    if preferred.exists():
        return preferred

    candidates = [path for path in result_root().iterdir() if path.is_dir() and list(path.glob("checkpoint-*.pt"))]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found under {result_root()}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def extract_checkpoint_epoch(path):
    match = re.search(r"checkpoint-(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def resolve_checkpoint_path(model_dir=None, checkpoint_epoch=None):
    search_root = result_root()
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()

    if checkpoint_epoch is not None:
        configured = model_dir / f"checkpoint-{checkpoint_epoch}.pt"
        if configured.exists():
            return configured

        matches = sorted(
            search_root.glob(f"**/checkpoint-{checkpoint_epoch}.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Could not find checkpoint-{checkpoint_epoch}.pt in {model_dir} or under {search_root}"
        )

    checkpoints = sorted(model_dir.glob("checkpoint-*.pt"), key=extract_checkpoint_epoch)
    if checkpoints:
        return checkpoints[-1]

    matches = sorted(
        search_root.glob("**/checkpoint-*.pt"),
        key=lambda path: (path.stat().st_mtime, extract_checkpoint_epoch(path)),
        reverse=True,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find any checkpoint under {search_root}")


def infer_model_num_groups(state_dict):
    group_embedding_key = "encoder.group_embedding.weight"
    if group_embedding_key in state_dict:
        return int(state_dict[group_embedding_key].size(0)) - 1
    return None


def calc_real_length(tour: Sequence[int], dist_matrix: torch.Tensor) -> float:
    return sum(
        float(dist_matrix[int(tour[i]), int(tour[(i + 1) % len(tour)])].item())
        for i in range(len(tour))
    )


def find_best_known_length(instance_file):
    tours_root = project_root() / "CTSPd(SOTA)" / "TOURS"
    instance_name = Path(instance_file).stem
    pattern = re.compile(rf"^{re.escape(instance_name)}\.(\d+(?:\.\d+)?)\.tour$")

    best_length = None
    for tour_file in tours_root.rglob(f"{instance_name}.*.tour"):
        match = pattern.match(tour_file.name)
        if match is None:
            continue
        length = float(match.group(1))
        if best_length is None or length < best_length:
            best_length = length

    return best_length


def save_tour_file(tour, instance_file, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_name = Path(instance_file).stem
    timestamp = datetime.now().strftime("%m%d_%H%M")
    tour_filename = f"{instance_name}.{timestamp}.tour"
    tour_path = output_dir / tour_filename

    lines = [
        f"NAME : {tour_filename}",
        f"COMMENT : Found by CTSP-d POMO Model {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        "TYPE : TOUR",
        f"DIMENSION : {len(tour)}",
        "TOUR_SECTION",
    ]
    lines.extend(str(node + 1) for node in tour)
    lines.append("-1")
    lines.append("EOF")

    with open(tour_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return tour_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CTSP-d cluster model on a single .ctspd instance.")
    parser.add_argument("--instance-file", default=str(default_instance_file()), help="Path to a .ctspd instance")
    parser.add_argument("--model-dir", default=None, help="Directory containing checkpoint-*.pt")
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to load. Defaults to the latest checkpoint in the resolved model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable 8-fold geometric augmentation for the reconstructed 2D features",
    )
    parser.add_argument(
        "--tour-dir",
        default=str(result_root() / "tours"),
        help="Directory for saving the generated .tour file",
    )
    parser.add_argument(
        "--no-save-tour",
        action="store_true",
        help="Skip saving the generated .tour file",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    instance_file = Path(args.instance_file)
    checkpoint_path = resolve_checkpoint_path(args.model_dir, args.checkpoint_epoch)

    print(f"Using device: {device}")
    print(f"Instance file: {instance_file}")
    print(f"Checkpoint: {checkpoint_path}")

    problems, raw_dist, relaxation_d, num_groups = parse_ctspd_file(str(instance_file))
    n_nodes = int(problems.size(1))
    print(f"Instance info: nodes={n_nodes}, d={relaxation_d}, groups={num_groups}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model_num_groups = infer_model_num_groups(state_dict)

    if model_num_groups is not None and num_groups > model_num_groups:
        raise ValueError(
            f"Instance uses {num_groups} groups, but checkpoint supports only {model_num_groups} groups."
        )

    model_params = dict(DEFAULT_MODEL_PARAMS)
    model_params["sqrt_embedding_dim"] = DEFAULT_MODEL_PARAMS["embedding_dim"] ** 0.5
    if model_num_groups is not None:
        model_params["num_groups"] = model_num_groups

    model = CTSPdModel(**model_params).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env = CTSPdEnv(
        problem_size=n_nodes,
        pomo_size=n_nodes,
        num_groups=num_groups,
        relaxation_d=relaxation_d,
    )

    aug_factor = 8 if args.augmentation else 1
    problems = problems.to(device)

    print(f"Running inference with aug_factor={aug_factor}...")
    if args.augmentation:
        print(
            "Warning: augmentation is applied to reconstructed 2D features, not the original "
            "distance matrix geometry."
        )
    with torch.no_grad():
        env.load_problems(1, aug_factor=aug_factor, problems=problems)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

    selected_node_list = env.selected_node_list
    if reward is None or selected_node_list is None:
        raise RuntimeError("Inference finished before producing a complete tour.")

    best_flat_idx = int(reward.argmax().item())
    reward_width = int(reward.size(1))
    best_batch_idx = best_flat_idx // reward_width
    best_pomo_idx = best_flat_idx % reward_width
    best_reward = float(reward[best_batch_idx, best_pomo_idx].item())
    best_tour = [int(node) for node in selected_node_list[best_batch_idx, best_pomo_idx].cpu().tolist()]
    real_length = calc_real_length(best_tour, raw_dist.cpu())

    best_known_length = find_best_known_length(instance_file)

    print("\n" + "=" * 50)
    print("Test Result")
    print("=" * 50)
    print(f"Model objective (normalized coordinates): {-best_reward:.4f}")
    print(f"Real tour length: {real_length:.2f}")
    print(f"Best tour indices: aug={best_batch_idx}, pomo={best_pomo_idx}")
    if best_known_length is not None:
        gap = ((real_length - best_known_length) / best_known_length) * 100.0
        print(f"Best known length: {best_known_length:.2f}")
        print(f"Gap: {gap:.2f}%")
    else:
        print("Best known length: not found in CTSPd(SOTA)/TOURS")
    print("=" * 50)

    if not args.no_save_tour:
        tour_path = save_tour_file(best_tour, instance_file, Path(args.tour_dir))
        print(f"Saved tour: {tour_path}")


if __name__ == "__main__":
    main()
