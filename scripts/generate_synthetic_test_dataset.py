import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CSTPd_cluster.CTSPd_ProblemDef import get_random_problems  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed synthetic CTSP-d test set for same-distribution evaluation."
    )
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--relaxation-d", type=int, default=1)
    parser.add_argument("--instance-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt file. Defaults to data/synthetic_tests/synthetic_n...pt",
    )
    return parser.parse_args()


def default_output(args: argparse.Namespace) -> Path:
    filename = (
        f"synthetic_n{args.problem_size}_g{args.num_groups}_d{args.relaxation_d}"
        f"_{args.instance_num}_seed{args.seed}.pt"
    )
    return PROJECT_ROOT / "data" / "synthetic_tests" / filename


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output(args)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    problems = get_random_problems(
        args.instance_num,
        args.problem_size,
        args.num_groups,
    ).cpu()

    metadata = {
        "dataset_type": "synthetic_fixed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "problem_size": args.problem_size,
        "num_groups": args.num_groups,
        "relaxation_d": args.relaxation_d,
        "instance_num": args.instance_num,
        "seed": args.seed,
        "notes": (
            "Fixed same-distribution synthetic test set: random coordinates and "
            "random priority-group assignment. This is the main n100/g8/d1 test set."
        ),
    }
    torch.save({"problems": problems, "metadata": metadata}, output_path)

    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved dataset: {output_path}")
    print(f"Saved metadata: {sidecar_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
