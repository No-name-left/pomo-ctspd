import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed synthetic CTSP-d instances to LKH format and run a "
            "LOW_FIRST-patched LKH executable as a classical benchmark."
        )
    )
    parser.add_argument(
        "--dataset-file",
        default="data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt",
        help="Fixed synthetic .pt dataset containing problems and metadata.",
    )
    parser.add_argument("--lkh-exe", default="LKH-3.0.14/LKH")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--max-trials", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--trace-level",
        type=int,
        default=0,
        help="LKH TRACE_LEVEL. Use 1 for detailed per-instance logs.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_dataset(dataset_file: Path, start_index: int, limit: Optional[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(dataset_file, map_location="cpu")
    problems = payload["problems"].detach().cpu()
    metadata = dict(payload.get("metadata", {}))
    end_index = None if limit is None else start_index + limit
    return problems[start_index:end_index], metadata


def make_distance_matrix(problem: torch.Tensor, scale: int) -> np.ndarray:
    coords = problem[:, :2].detach().cpu().numpy().astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    matrix = np.rint(dist * scale).astype(np.int64)
    np.fill_diagonal(matrix, 999_999_999)
    return matrix


def write_ctspd_instance(
    path: Path,
    instance_id: str,
    problem: torch.Tensor,
    matrix: np.ndarray,
    num_groups: int,
    relaxation_d: int,
) -> None:
    priorities = problem[:, 2].detach().cpu().numpy().astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NAME : {instance_id}\n")
        f.write("TYPE : CTSP-D\n")
        f.write(f"DIMENSION : {matrix.shape[0]}\n")
        f.write(f"GROUPS : {num_groups}\n")
        f.write(f"RELAXATION_LEVEL : {relaxation_d}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            f.write(" ".join(str(int(value)) for value in row) + "\n")
        f.write("GROUP_SECTION\n")
        for group in range(1, num_groups + 1):
            nodes = [str(idx + 1) for idx, value in enumerate(priorities) if int(value) == group]
            f.write(f"{group} {' '.join(nodes)} -1\n")
        f.write("EOF\n")


def write_parameter_file(
    path: Path,
    problem_file: Path,
    tour_pattern: Path,
    args: argparse.Namespace,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"PROBLEM_FILE = {problem_file}",
        "SPECIAL",
        f"MAX_CANDIDATES = {args.max_candidates}",
        f"TRACE_LEVEL = {args.trace_level}",
        f"RUNS = {args.runs}",
        f"SEED = {seed}",
        f"MAX_TRIALS = {args.max_trials}",
        f"TOUR_FILE = {tour_pattern}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_tour(path: Path) -> list[int]:
    nodes = []
    in_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "TOUR_SECTION":
            in_section = True
            continue
        if not in_section:
            continue
        if line == "-1" or line == "EOF":
            break
        if line:
            nodes.append(int(line) - 1)
    return nodes


def tour_cost_int(tour: list[int], matrix: np.ndarray) -> int:
    arr = np.asarray(tour, dtype=np.int64)
    return int(matrix[arr, np.roll(arr, -1)].sum())


def feasibility_violations(tour: list[int], priorities: np.ndarray, relaxation_d: int) -> int:
    n_nodes = len(priorities)
    unvisited = set(range(n_nodes))
    remaining = Counter(int(value) for value in priorities)
    violations = 0
    for node in tour:
        if node not in unvisited:
            violations += 1
            continue
        current_min = min(group for group, count in remaining.items() if count > 0)
        group = int(priorities[node])
        if group > current_min + relaxation_d:
            violations += 1
        remaining[group] -= 1
        unvisited.remove(node)
    if len(tour) != n_nodes or unvisited:
        violations += len(unvisited)
    return violations


def best_feasible_orientation(
    tour: list[int],
    priorities: np.ndarray,
    relaxation_d: int,
) -> tuple[list[int], int, str, int]:
    best_tour = tour
    best_violations = math.inf
    best_direction = "forward"
    best_rotation = 0
    for direction, candidate in (("forward", tour), ("reverse", list(reversed(tour)))):
        for rotation in range(len(candidate)):
            rotated = candidate[rotation:] + candidate[:rotation]
            violations = feasibility_violations(rotated, priorities, relaxation_d)
            if violations < best_violations:
                best_tour = rotated
                best_violations = violations
                best_direction = direction
                best_rotation = rotation
                if violations == 0:
                    return best_tour, violations, best_direction, best_rotation
    return best_tour, int(best_violations), best_direction, best_rotation


def write_tour(path: Path, instance_id: str, tour: list[int], cost: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"NAME : {instance_id}.lkh_normalized.tour",
        f"COMMENT : Length = {cost:.9f}",
        "COMMENT : Rotated/reversed, if needed, to match LOW_FIRST CTSP-d feasibility.",
        "TYPE : TOUR",
        f"DIMENSION : {len(tour)}",
        "TOUR_SECTION",
    ]
    lines.extend(str(node + 1) for node in tour)
    lines.extend(["-1", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_cost_min(output: str) -> Optional[int]:
    match = re.search(r"Cost\.min\s*=\s*([0-9]+)", output)
    return int(match.group(1)) if match else None


def run_lkh_instance(
    lkh_exe: Path,
    par_file: Path,
    raw_tour_dir: Path,
    tour_glob: str,
    timeout_sec: float,
) -> tuple[subprocess.CompletedProcess[str], list[Path], float]:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(lkh_exe), str(par_file)],
        cwd=str(lkh_exe.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.perf_counter() - start
    tours = sorted(raw_tour_dir.glob(tour_glob))
    return completed, tours, elapsed


def make_output_dir(output_dir: Optional[str]) -> Path:
    if output_dir:
        path = resolve_path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "test_results" / f"{timestamp}_lkh_low_first_synthetic"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    dataset_file = resolve_path(args.dataset_file)
    lkh_exe = resolve_path(args.lkh_exe)
    if not lkh_exe.exists():
        raise FileNotFoundError(f"LKH executable not found: {lkh_exe}")

    output_dir = make_output_dir(args.output_dir)
    problem_dir = output_dir / "instances"
    par_dir = output_dir / "par"
    raw_tour_dir = output_dir / "raw_tours"
    normalized_tour_dir = output_dir / "normalized_tours"
    raw_tour_dir.mkdir(parents=True, exist_ok=True)

    problems, metadata = load_dataset(dataset_file, args.start_index, args.limit)
    num_groups = int(metadata.get("num_groups", int(problems[:, :, 2].max().item())))
    relaxation_d = int(metadata.get("relaxation_d", 1))

    rows = []
    for local_idx, problem in enumerate(problems):
        dataset_idx = args.start_index + local_idx
        instance_id = f"synthetic_{dataset_idx:06d}"
        matrix = make_distance_matrix(problem, args.scale)
        instance_file = problem_dir / f"{instance_id}.ctspd"
        par_file = par_dir / f"{instance_id}.par"
        tour_pattern = raw_tour_dir / f"{instance_id}.$.tour"
        for old_tour in raw_tour_dir.glob(f"{instance_id}.*.tour"):
            old_tour.unlink()

        write_ctspd_instance(instance_file, instance_id, problem, matrix, num_groups, relaxation_d)
        write_parameter_file(par_file, instance_file, tour_pattern, args, args.seed + dataset_idx)
        completed, tour_files, elapsed = run_lkh_instance(
            lkh_exe,
            par_file,
            raw_tour_dir,
            f"{instance_id}.*.tour",
            args.timeout_sec,
        )
        output_file = output_dir / "logs" / f"{instance_id}.log"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(completed.stdout, encoding="utf-8")

        lkh_cost_min = extract_cost_min(completed.stdout)
        raw_tour_path = tour_files[-1] if tour_files else None
        normalized_tour_path = None
        feasible = False
        violations = None
        cost_int = lkh_cost_min
        direction = None
        rotation = None
        if raw_tour_path is not None:
            raw_tour = parse_tour(raw_tour_path)
            priorities = problem[:, 2].detach().cpu().numpy().astype(np.int64)
            normalized_tour, violations, direction, rotation = best_feasible_orientation(
                raw_tour,
                priorities,
                relaxation_d,
            )
            feasible = violations == 0
            cost_int = tour_cost_int(normalized_tour, matrix)
            normalized_tour_path = normalized_tour_dir / f"{instance_id}.{cost_int}.tour"
            write_tour(normalized_tour_path, instance_id, normalized_tour, cost_int / args.scale)

        row = {
            "instance_id": instance_id,
            "dataset_index": dataset_idx,
            "problem_size": int(problem.size(0)),
            "num_groups": num_groups,
            "relaxation_d": relaxation_d,
            "scale": args.scale,
            "lkh_cost_int": cost_int,
            "lkh_cost": None if cost_int is None else cost_int / args.scale,
            "is_feasible": feasible,
            "violation_count": violations,
            "orientation": direction,
            "rotation": rotation,
            "return_code": completed.returncode,
            "time_sec": elapsed,
            "instance_file": str(instance_file),
            "parameter_file": str(par_file),
            "raw_tour_file": str(raw_tour_path) if raw_tour_path else None,
            "normalized_tour_file": str(normalized_tour_path) if normalized_tour_path else None,
            "log_file": str(output_file),
        }
        rows.append(row)
        print(
            "[{}/{}] {} cost={} feasible={} time={:.3f}s".format(
                local_idx + 1,
                len(problems),
                instance_id,
                row["lkh_cost"],
                row["is_feasible"],
                elapsed,
            )
        )

    costs = [float(row["lkh_cost"]) for row in rows if row["lkh_cost"] is not None]
    feasible_flags = [1.0 if row["is_feasible"] else 0.0 for row in rows]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": str(dataset_file),
        "dataset_metadata": metadata,
        "lkh_executable": str(lkh_exe),
        "output_dir": str(output_dir),
        "test_instance_num": len(rows),
        "start_index": args.start_index,
        "limit": args.limit,
        "scale": args.scale,
        "max_trials": args.max_trials,
        "runs": args.runs,
        "seed": args.seed,
        "average_cost": float(mean(costs)) if costs else None,
        "std_cost": float(pstdev(costs)) if len(costs) > 1 else 0.0 if costs else None,
        "best_cost": min(costs) if costs else None,
        "worst_cost": max(costs) if costs else None,
        "feasible_rate": float(mean(feasible_flags)) if feasible_flags else None,
        "total_time_sec": float(sum(float(row["time_sec"]) for row in rows)),
    }
    write_csv(output_dir / "lkh_instances.csv", rows)
    (output_dir / "lkh_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {output_dir / 'lkh_instances.csv'}")
    print(f"Saved: {output_dir / 'lkh_summary.json'}")


if __name__ == "__main__":
    main()
