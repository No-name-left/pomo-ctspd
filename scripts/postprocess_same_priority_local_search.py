import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CSTPd_cluster.CTSPd_ProblemDef import load_ctspd_tour, parse_ctspd_file  # noqa: E402
from scripts.compare_sota_instances import (  # noqa: E402
    calc_feasibility,
    pct_gap,
    pct_improvement,
    save_tour,
    winner_name,
)


DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "comparison_results"
    / "sota_cluster_large_n100_d1__bsl_vs_cluster__enhanced_inference_mds_aug_sample64__10_instances"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve generated CTSP-d tours by swapping nodes with the same priority."
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Comparison result folder containing instances.csv and generated tours.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Defaults to <source-dir>__same_priority_swap_ls<PASSES>.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=20,
        help="Maximum best-improvement swap passes per tour.",
    )
    return parser.parse_args()


def tour_cost(tour: list[int] | np.ndarray, dist: np.ndarray) -> float:
    arr = np.asarray(tour, dtype=np.int64)
    return float(dist[arr, np.roll(arr, -1)].sum())


def same_priority_swap_search(
    tour: list[int],
    dist: np.ndarray,
    priorities: np.ndarray,
    max_passes: int,
) -> tuple[list[int], float, int]:
    best = np.asarray(tour, dtype=np.int64)
    best_cost = tour_cost(best, dist)
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
            cost = tour_cost(swapped, dist)
            if cost < candidate_cost - 1e-9:
                candidate = swapped
                candidate_cost = cost

        if candidate is None:
            break

        best = candidate
        best_cost = candidate_cost
        pass_count += 1

    return best.tolist(), best_cost, pass_count


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            values.append(float(value))
    return values


def stats(rows: list[dict[str, Any]], key: str) -> dict[str, Optional[float]]:
    values = numeric_values(rows, key)
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    winners = [row["winner"] for row in rows]
    return {
        "instance_count": len(rows),
        "bsl_cost": stats(rows, "bsl_cost"),
        "cluster_cost": stats(rows, "cluster_cost"),
        "lkh_cost": stats(rows, "lkh_cost"),
        "bsl_gap_to_lkh_percent": stats(rows, "bsl_gap_to_lkh_percent"),
        "cluster_gap_to_lkh_percent": stats(rows, "cluster_gap_to_lkh_percent"),
        "cluster_minus_bsl": stats(rows, "cluster_minus_bsl"),
        "cluster_improvement_vs_bsl_percent": stats(rows, "cluster_improvement_vs_bsl_percent"),
        "bsl_local_search_improvement": stats(rows, "bsl_local_search_improvement"),
        "cluster_local_search_improvement": stats(rows, "cluster_local_search_improvement"),
        "cluster_win_count": winners.count("cluster"),
        "bsl_win_count": winners.count("bsl"),
        "tie_count": winners.count("tie"),
        "cluster_win_rate": winners.count("cluster") / len(rows) if rows else None,
        "all_bsl_feasible": all(str(row["bsl_feasible"]).lower() == "true" for row in rows),
        "all_cluster_feasible": all(str(row["cluster_feasible"]).lower() == "true" for row in rows),
    }


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    input_csv = source_dir / "instances.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing instances.csv: {input_csv}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif source_dir.name.endswith("__10_instances"):
        base_name = source_dir.name.removesuffix("__10_instances")
        output_dir = (
            source_dir.parent
            / f"{base_name}_plus_same_priority_ls{args.passes}__10_instances"
        )
    else:
        output_dir = source_dir.parent / f"{source_dir.name}__same_priority_swap_ls{args.passes}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tour_dir = output_dir / "tours"

    rows = list(csv.DictReader(input_csv.open(encoding="utf-8")))
    output_rows: list[dict[str, Any]] = []

    for row in rows:
        instance_path = PROJECT_ROOT / "CTSPd(SOTA)" / "INSTANCES" / row["category"] / row["instance"]
        problems, raw_dist, relaxation_d, _ = parse_ctspd_file(str(instance_path))
        dist = raw_dist.detach().cpu().numpy()
        priorities = problems[0, :, 2].detach().cpu().numpy().astype(np.int64)

        output_row = dict(row)
        for model_name in ("bsl", "cluster"):
            source_tour_file = Path(row[f"{model_name}_tour_file"])
            tour = load_ctspd_tour(str(source_tour_file))
            if tour is None:
                raise ValueError(f"Could not parse tour: {source_tour_file}")

            before_cost = tour_cost(tour, dist)
            improved_tour, improved_cost, pass_count = same_priority_swap_search(
                tour,
                dist,
                priorities,
                args.passes,
            )
            violations, feasible = calc_feasibility(
                improved_tour,
                problems[0, :, 2].detach().cpu(),
                relaxation_d,
            )
            improved_tour_file = save_tour(
                improved_tour,
                improved_cost,
                instance_path,
                f"{model_name}_same_priority_ls",
                tour_dir,
            )

            output_row[f"{model_name}_cost_before_local_search"] = before_cost
            output_row[f"{model_name}_local_search_improvement"] = before_cost - improved_cost
            output_row[f"{model_name}_local_search_passes"] = pass_count
            output_row[f"{model_name}_source_tour_file"] = str(source_tour_file)
            output_row[f"{model_name}_tour_file"] = str(improved_tour_file)
            output_row[f"{model_name}_cost"] = improved_cost
            output_row[f"{model_name}_violation_count"] = violations
            output_row[f"{model_name}_feasible"] = feasible

        lkh_cost = float(output_row["lkh_cost"]) if output_row.get("lkh_cost") not in (None, "") else None
        bsl_cost = float(output_row["bsl_cost"])
        cluster_cost = float(output_row["cluster_cost"])
        output_row["bsl_gap_to_lkh_percent"] = pct_gap(bsl_cost, lkh_cost)
        output_row["cluster_gap_to_lkh_percent"] = pct_gap(cluster_cost, lkh_cost)
        output_row["cluster_minus_bsl"] = cluster_cost - bsl_cost
        output_row["cluster_improvement_vs_bsl_percent"] = pct_improvement(bsl_cost, cluster_cost)
        output_row["winner"] = winner_name(bsl_cost, cluster_cost)
        output_rows.append(output_row)

        print(
            "{}: bsl {:.0f}->{:.0f}, cluster {:.0f}->{:.0f}, winner={}".format(
                row["instance"],
                float(row["bsl_cost"]),
                bsl_cost,
                float(row["cluster_cost"]),
                cluster_cost,
                output_row["winner"],
            )
        )

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with (output_dir / "instances.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "local_search": {
            "type": "same_priority_swap",
            "passes": args.passes,
            "note": "Only swaps nodes occupying positions with the same priority, so the priority sequence is unchanged.",
        },
        "summary": summarize(output_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nOutput: {output_dir}")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
