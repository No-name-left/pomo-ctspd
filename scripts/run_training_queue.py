import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_PRIORITY_GROUP_QUEUE = [
    "CSTPd_cluster/POMO/train_n100.py",
    "CSTPd_cluster/POMO/train_n100_wo_group_embedding.py",
    "CSTPd_cluster/POMO/train_n100_wo_fusion_gate.py",
    "CSTPd_cluster/POMO/train_n100_wo_cluster_bias.py",
    "CSTPd_cluster/POMO/train_n100_wo_priority_distance_bias.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training scripts sequentially and keep queue logs.")
    parser.add_argument(
        "--preset",
        default="main_priority_group",
        choices=["main_priority_group", "custom"],
    )
    parser.add_argument("--scripts", nargs="*", default=None)
    parser.add_argument("--run-root", default="training_runs")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def resolve_scripts(args: argparse.Namespace) -> list[Path]:
    raw_scripts = args.scripts
    if raw_scripts is None:
        if args.preset == "main_priority_group":
            raw_scripts = MAIN_PRIORITY_GROUP_QUEUE
        else:
            raise ValueError("--scripts is required when --preset custom is used.")
    scripts = []
    for raw in raw_scripts:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Training script not found: {path}")
        scripts.append(path)
    return scripts


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scripts = resolve_scripts(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_dir = run_root / f"{timestamp}_{args.preset}_queue"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "queue_state.json"
    state: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "python": args.python,
        "preset": args.preset,
        "scripts": [str(path.relative_to(PROJECT_ROOT)) for path in scripts],
        "jobs": [],
        "status": "running",
    }
    write_state(state_path, state)

    for script in scripts:
        rel_script = script.relative_to(PROJECT_ROOT)
        log_name = str(rel_script).replace("/", "__").replace(".py", ".log")
        log_path = run_dir / log_name
        job = {
            "script": str(rel_script),
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }
        state["jobs"].append(job)
        write_state(state_path, state)

        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"[queue] starting {rel_script} at {job['started_at']}\n")
            log_file.flush()
            completed = subprocess.run(
                [args.python, str(script)],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job["elapsed_time_sec"] = time.time() - start
        job["returncode"] = completed.returncode
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        write_state(state_path, state)

        if completed.returncode != 0 and args.stop_on_failure:
            state["status"] = "failed"
            write_state(state_path, state)
            raise SystemExit(completed.returncode)

    state["status"] = "completed"
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_state(state_path, state)
    print(f"Training queue completed: {run_dir}")


if __name__ == "__main__":
    main()
