import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FIELDS = [
    "experiment_dir",
    "model_type",
    "model_variant",
    "dataset_type",
    "problem_size",
    "num_groups",
    "relaxation_d",
    "checkpoint_path",
    "average_cost",
    "std_cost",
    "average_gap_to_lkh_percent",
    "median_gap_to_lkh_percent",
    "inference_time_per_instance_sec",
    "total_inference_time_sec",
    "augmentation_factor",
    "feature_modes",
    "sampling_runs",
    "sampling_temperatures",
    "sampling_top_ks",
    "same_priority_ls_passes",
    "average_cost_before_local_search",
    "average_local_search_improvement",
    "total_training_time_sec",
    "best_epoch",
    "best_score",
    "test_instance_num",
    "feasible_rate",
    "device",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect test_summary.json files into one paper-friendly summary.csv."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["test_results"],
        help="Experiment dirs, parent dirs, or test_summary.json files.",
    )
    parser.add_argument("--output", default="summary.csv")
    return parser.parse_args()


def find_summary_files(inputs: list[str]) -> list[Path]:
    files = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file() and path.name == "test_summary.json":
            files.append(path)
        elif path.is_dir():
            direct = path / "test_summary.json"
            if direct.exists():
                files.append(direct)
            files.extend(path.rglob("test_summary.json"))
    return sorted(set(files))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_training_progress(checkpoint_path: Any) -> dict[str, Any]:
    if not checkpoint_path:
        return {}
    progress_path = Path(str(checkpoint_path)).parent / "training_progress.json"
    if not progress_path.exists():
        return {}
    try:
        return read_json(progress_path)
    except json.JSONDecodeError:
        return {}


def normalize_value(value: Any) -> Any:
    if value is None:
        return math.nan
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def make_row(summary: dict[str, Any]) -> dict[str, Any]:
    training_progress = read_training_progress(summary.get("checkpoint_path"))
    row = {field: normalize_value(summary.get(field)) for field in SUMMARY_FIELDS}
    row["total_training_time_sec"] = normalize_value(
        training_progress.get("total_training_time_sec", row.get("total_training_time_sec"))
    )
    row["best_epoch"] = normalize_value(training_progress.get("best_epoch", row.get("best_epoch")))
    row["best_score"] = normalize_value(training_progress.get("best_value", row.get("best_score")))
    return row


def main() -> None:
    args = parse_args()
    summary_files = find_summary_files(args.inputs)
    rows = [make_row(read_json(path)) for path in summary_files]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found summaries: {len(summary_files)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
