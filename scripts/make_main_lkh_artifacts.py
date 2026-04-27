import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = [
    {
        "key": "lkh_low_first",
        "label": "LKH",
        "kind": "classical",
        "color": "#3A3A3A",
        "metric_field": "lkh_cost",
        "summary_file": "lkh_summary.json",
        "instances_file": "lkh_instances.csv",
        "training_metrics": None,
    },
    {
        "key": "baseline",
        "label": "POMO baseline",
        "kind": "neural",
        "color": "#4C78A8",
        "metric_field": "model_cost",
        "summary_file": "test_summary.json",
        "instances_file": "test_instances.csv",
        "training_metrics": "CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/training_metrics.csv",
    },
    {
        "key": "full_learnable",
        "label": "Full model",
        "kind": "neural",
        "color": "#54A24B",
        "metric_field": "model_cost",
        "summary_file": "test_summary.json",
        "instances_file": "test_instances.csv",
        "training_metrics": "CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias/training_metrics.csv",
    },
    {
        "key": "scheduled_bias",
        "label": "Scheduled bias",
        "kind": "neural",
        "color": "#F58518",
        "metric_field": "model_cost",
        "summary_file": "test_summary.json",
        "instances_file": "test_instances.csv",
        "training_metrics": "CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1/training_metrics.csv",
    },
    {
        "key": "wo_all_bias",
        "label": "w/o all bias",
        "kind": "neural",
        "color": "#E45756",
        "metric_field": "model_cost",
        "summary_file": "test_summary.json",
        "instances_file": "test_instances.csv",
        "training_metrics": "CSTPd_cluster/POMO/result/26日_10点20分_cluster_n100_d1_wo_all_bias/training_metrics.csv",
    },
    {
        "key": "wo_fusion_gate",
        "label": "w/o fusion gate",
        "kind": "neural",
        "color": "#72B7B2",
        "metric_field": "model_cost",
        "summary_file": "test_summary.json",
        "instances_file": "test_instances.csv",
        "training_metrics": "CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1/training_metrics.csv",
    },
    {
        "key": "wo_group_embedding",
        "label": "w/o group emb.",
        "kind": "neural",
        "color": "#B279A2",
        "metric_field": "model_cost",
        "summary_file": "test_summary.json",
        "instances_file": "test_instances.csv",
        "training_metrics": "CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1/training_metrics.csv",
    },
]

EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper tables and figures for the synthetic CTSP-d main experiment with LKH."
    )
    parser.add_argument(
        "--result-dir",
        default="test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427",
        help="Main experiment directory containing evaluations/<model>/ outputs.",
    )
    parser.add_argument(
        "--output-subdir",
        default="paper_artifacts",
        help="Subdirectory under result-dir for paper-ready artifacts.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ci95(values: list[float]) -> float:
    if not values:
        return math.nan
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return 1.96 * std / math.sqrt(len(values))


def mean_or_nan(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def load_model_outputs(result_dir: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing = []
    for config in MODEL_ORDER:
        eval_dir = result_dir / "evaluations" / config["key"]
        summary_path = eval_dir / str(config["summary_file"])
        instances_path = eval_dir / str(config["instances_file"])
        if not summary_path.exists() or not instances_path.exists():
            missing.append(str(eval_dir))
            continue
        summary = read_json(summary_path)
        rows = read_csv_rows(instances_path)
        costs: dict[str, float] = {}
        feasible: dict[str, bool] = {}
        times: dict[str, float] = {}
        for row in rows:
            instance_id = row["instance_id"]
            cost = to_float(row.get(str(config["metric_field"])))
            if cost is None:
                continue
            costs[instance_id] = cost
            feasible[instance_id] = to_bool(row.get("is_feasible", ""))
            time_value = to_float(row.get("inference_time_sec") or row.get("time_sec"))
            if time_value is not None:
                times[instance_id] = time_value
        loaded[config["key"]] = {
            "config": config,
            "summary": summary,
            "costs": costs,
            "feasible": feasible,
            "times": times,
            "summary_path": summary_path,
            "instances_path": instances_path,
        }
    if missing:
        raise FileNotFoundError("Missing evaluation outputs:\n" + "\n".join(missing))
    return loaded


def common_instance_ids(loaded: dict[str, dict[str, Any]]) -> list[str]:
    id_sets = [set(item["costs"]) for item in loaded.values()]
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise RuntimeError("No common instance ids across model outputs.")
    return common


def build_per_instance_rows(
    loaded: dict[str, dict[str, Any]], ids: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id in ids:
        row: dict[str, Any] = {"instance_id": instance_id}
        values = {key: loaded[key]["costs"][instance_id] for key in loaded}
        best_key = min(values, key=values.get)
        lkh_cost = values["lkh_low_first"]
        full_cost = values["full_learnable"]
        row["best_method"] = best_key
        for config in MODEL_ORDER:
            key = config["key"]
            cost = values[key]
            row[f"{key}_cost"] = cost
            row[f"{key}_gap_to_lkh_percent"] = (cost - lkh_cost) / lkh_cost * 100.0
            row[f"{key}_gap_to_full_percent"] = (cost - full_cost) / full_cost * 100.0
        rows.append(row)
    return rows


def build_summary_rows(
    loaded: dict[str, dict[str, Any]], ids: list[str]
) -> list[dict[str, Any]]:
    lkh_values = [loaded["lkh_low_first"]["costs"][instance_id] for instance_id in ids]
    full_values = [loaded["full_learnable"]["costs"][instance_id] for instance_id in ids]
    baseline_values = [loaded["baseline"]["costs"][instance_id] for instance_id in ids]
    rows: list[dict[str, Any]] = []
    for config in MODEL_ORDER:
        key = config["key"]
        values = [loaded[key]["costs"][instance_id] for instance_id in ids]
        gaps_lkh = [(value - lkh) / lkh * 100.0 for value, lkh in zip(values, lkh_values)]
        gaps_full = [(value - full) / full * 100.0 for value, full in zip(values, full_values)]
        gaps_baseline = [(value - base) / base * 100.0 for value, base in zip(values, baseline_values)]
        feasible = [loaded[key]["feasible"].get(instance_id, False) for instance_id in ids]
        times = [loaded[key]["times"][instance_id] for instance_id in ids if instance_id in loaded[key]["times"]]
        summary = loaded[key]["summary"]
        total_time = summary.get("total_time_sec", summary.get("total_inference_time_sec"))
        time_per_instance = (
            float(total_time) / len(ids)
            if total_time is not None
            else mean_or_nan(times)
        )
        rows.append(
            {
                "model_key": key,
                "model_label": config["label"],
                "method_type": config["kind"],
                "common_instance_num": len(ids),
                "average_cost": mean_or_nan(values),
                "std_cost": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "ci95_cost": ci95(values),
                "best_cost": min(values),
                "worst_cost": max(values),
                "average_gap_to_lkh_percent": mean_or_nan(gaps_lkh),
                "median_gap_to_lkh_percent": median_or_nan(gaps_lkh),
                "average_gap_to_full_percent": mean_or_nan(gaps_full),
                "average_gap_to_baseline_percent": mean_or_nan(gaps_baseline),
                "wins_vs_lkh": sum(1 for value, lkh in zip(values, lkh_values) if value < lkh - EPS),
                "ties_vs_lkh": sum(1 for value, lkh in zip(values, lkh_values) if abs(value - lkh) <= EPS),
                "losses_vs_lkh": sum(1 for value, lkh in zip(values, lkh_values) if value > lkh + EPS),
                "feasible_rate": mean_or_nan([1.0 if item else 0.0 for item in feasible]),
                "time_per_instance_sec": time_per_instance,
                "total_time_sec": total_time,
                "speedup_vs_lkh": "",
                "augmentation_factor": summary.get("augmentation_factor", ""),
                "checkpoint_path": summary.get("checkpoint_path", ""),
                "lkh_max_trials": summary.get("max_trials", ""),
                "lkh_runs": summary.get("runs", ""),
                "lkh_scale": summary.get("scale", ""),
            }
        )
    lkh_time = next(
        float(row["time_per_instance_sec"])
        for row in rows
        if row["model_key"] == "lkh_low_first"
    )
    for row in rows:
        method_time = float(row["time_per_instance_sec"])
        row["speedup_vs_lkh"] = lkh_time / method_time if method_time > 0 else math.nan
    return rows


def build_pairwise_tables(
    loaded: dict[str, dict[str, Any]], ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    keys = [config["key"] for config in MODEL_ORDER]
    counts = np.zeros((len(keys), len(keys)), dtype=np.int64)
    percents = np.zeros((len(keys), len(keys)), dtype=np.float64)
    for i, row_key in enumerate(keys):
        for j, col_key in enumerate(keys):
            if i == j:
                counts[i, j] = 0
                percents[i, j] = 50.0
                continue
            wins = sum(
                1
                for instance_id in ids
                if loaded[row_key]["costs"][instance_id] < loaded[col_key]["costs"][instance_id] - EPS
            )
            counts[i, j] = wins
            percents[i, j] = wins / len(ids) * 100.0

    count_rows: list[dict[str, Any]] = []
    percent_rows: list[dict[str, Any]] = []
    for i, config in enumerate(MODEL_ORDER):
        count_row: dict[str, Any] = {"model_key": config["key"], "model_label": config["label"]}
        percent_row: dict[str, Any] = {"model_key": config["key"], "model_label": config["label"]}
        for j, other in enumerate(MODEL_ORDER):
            count_row[f"beats_{other['key']}"] = int(counts[i, j])
            percent_row[f"beats_{other['key']}_percent"] = float(percents[i, j])
        count_rows.append(count_row)
        percent_rows.append(percent_row)
    return count_rows, percent_rows, percents


def write_main_tables(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    pair_count_rows: list[dict[str, Any]],
    pair_percent_rows: list[dict[str, Any]],
    per_instance_rows: list[dict[str, Any]],
) -> None:
    write_csv(out_dir.parent / "summary.csv", summary_rows)
    write_csv(out_dir / "main_results_table.csv", summary_rows)
    write_csv(out_dir / "pairwise_win_counts.csv", pair_count_rows)
    write_csv(out_dir / "pairwise_win_percent.csv", pair_percent_rows)
    write_csv(out_dir / "per_instance_costs.csv", per_instance_rows)

    main_md_rows = []
    for row in summary_rows:
        main_md_rows.append(
            [
                row["model_label"],
                fmt(row["average_cost"], 4),
                fmt(row["ci95_cost"], 4),
                fmt(row["average_gap_to_lkh_percent"], 3),
                fmt(row["average_gap_to_full_percent"], 3),
                fmt(100.0 * float(row["feasible_rate"]), 1),
                fmt(row["time_per_instance_sec"], 3),
                fmt(row["speedup_vs_lkh"], 1),
            ]
        )
    (out_dir / "main_results_table.md").write_text(
        markdown_table(
            [
                "Method",
                "Avg. cost",
                "95% CI",
                "Gap to LKH (%)",
                "Gap to full (%)",
                "Feasible (%)",
                "Time/inst. (s)",
                "Speedup vs LKH",
            ],
            main_md_rows,
        ),
        encoding="utf-8",
    )

    full_row = next(row for row in summary_rows if row["model_key"] == "full_learnable")
    ablation_rows = []
    for row in summary_rows:
        if row["method_type"] != "neural":
            continue
        delta = float(row["average_cost"]) - float(full_row["average_cost"])
        ablation_rows.append(
            {
                "model_key": row["model_key"],
                "model_label": row["model_label"],
                "average_cost": row["average_cost"],
                "delta_vs_full_cost": delta,
                "average_gap_to_full_percent": row["average_gap_to_full_percent"],
                "average_gap_to_lkh_percent": row["average_gap_to_lkh_percent"],
                "feasible_rate": row["feasible_rate"],
                "time_per_instance_sec": row["time_per_instance_sec"],
            }
        )
    write_csv(out_dir / "ablation_table.csv", ablation_rows)
    ablation_md_rows = [
        [
            row["model_label"],
            fmt(row["average_cost"], 4),
            fmt(row["delta_vs_full_cost"], 4),
            fmt(row["average_gap_to_full_percent"], 3),
            fmt(row["average_gap_to_lkh_percent"], 3),
        ]
        for row in ablation_rows
    ]
    (out_dir / "ablation_table.md").write_text(
        markdown_table(
            ["Model", "Avg. cost", "Delta vs full", "Gap to full (%)", "Gap to LKH (%)"],
            ablation_md_rows,
        ),
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_average_cost(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [row["model_label"] for row in summary_rows]
    means = [float(row["average_cost"]) for row in summary_rows]
    errors = [float(row["ci95_cost"]) for row in summary_rows]
    colors = [next(config["color"] for config in MODEL_ORDER if config["key"] == row["model_key"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Average tour length")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Main Results on Synthetic CTSP-d")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, out_dir, "average_cost_with_lkh_bar")


def plot_gap_to_lkh(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [row["model_label"] for row in summary_rows]
    gaps = [float(row["average_gap_to_lkh_percent"]) for row in summary_rows]
    colors = [next(config["color"] for config in MODEL_ORDER if config["key"] == row["model_key"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(labels))
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.bar(x, gaps, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Gap to LKH (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Average Per-instance Gap to LKH")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, out_dir, "gap_to_lkh_bar")


def plot_gap_boxplot(
    per_instance_rows: list[dict[str, Any]], out_dir: Path
) -> None:
    neural_configs = [config for config in MODEL_ORDER if config["kind"] == "neural"]
    labels = [config["label"] for config in neural_configs]
    data = [
        [float(row[f"{config['key']}_gap_to_lkh_percent"]) for row in per_instance_rows]
        for config in neural_configs
    ]
    colors = [config["color"] for config in neural_configs]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#222222")
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.2)
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_ylabel("Gap to LKH (%)")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Per-instance Gap Distribution")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, out_dir, "per_instance_gap_to_lkh_boxplot")


def plot_pairwise_heatmap(win_percent: np.ndarray, out_dir: Path) -> None:
    labels = [config["label"] for config in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(win_percent, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Pairwise Win Rate")
    ax.set_xlabel("Compared method")
    ax.set_ylabel("Method")
    for i in range(win_percent.shape[0]):
        for j in range(win_percent.shape[1]):
            text_color = "white" if win_percent[i, j] >= 60 else "#222222"
            ax.text(j, i, f"{win_percent[i, j]:.0f}", ha="center", va="center", color=text_color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row method wins (%)")
    save_figure(fig, out_dir, "pairwise_win_heatmap")


def plot_runtime(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [row["model_label"] for row in summary_rows]
    times = [float(row["time_per_instance_sec"]) for row in summary_rows]
    colors = [next(config["color"] for config in MODEL_ORDER if config["key"] == row["model_key"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, times, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Time per instance (s, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Inference Time Comparison")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7, which="both")
    ax.set_axisbelow(True)
    save_figure(fig, out_dir, "time_per_instance_bar_log")


def plot_training_curves(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    plotted = False
    for config in MODEL_ORDER:
        metrics_path = config.get("training_metrics")
        if not metrics_path:
            continue
        path = resolve_path(str(metrics_path))
        if not path.exists():
            continue
        rows = read_csv_rows(path)
        epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch") and row.get("train_score")]
        scores = [float(row["train_score"]) for row in rows if row.get("epoch") and row.get("train_score")]
        if not epochs:
            continue
        ax.plot(epochs, scores, label=config["label"], color=config["color"], linewidth=1.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training score")
    ax.set_title("Training Score Curves")
    ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save_figure(fig, out_dir, "training_score_curves")


def write_readme(out_dir: Path, result_dir: Path, common_n: int) -> None:
    text = f"""# Paper Artifacts

Generated from `{result_dir}` using `{Path(__file__).name}`.

Common instances across all methods: {common_n}.

Files:
- `main_results_table.csv` / `.md`: compact table for the main synthetic experiment.
- `ablation_table.csv` / `.md`: neural-model ablation table using the full model as reference.
- `per_instance_costs.csv`: aligned per-instance costs and gaps.
- `pairwise_win_counts.csv` and `pairwise_win_percent.csv`: pairwise comparison tables.
- `average_cost_with_lkh_bar.*`: average cost with 95% confidence intervals.
- `gap_to_lkh_bar.*`: average per-instance gap to LKH.
- `per_instance_gap_to_lkh_boxplot.*`: distribution of per-instance gaps to LKH.
- `pairwise_win_heatmap.*`: pairwise win-rate heatmap.
- `time_per_instance_bar_log.*`: inference-time comparison on a log scale.
- `training_score_curves.*`: training-score curves for neural models.

LKH here refers to the LOW_FIRST-patched CTSP-d build used for the synthetic data convention.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = resolve_path(args.result_dir)
    out_dir = result_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model_outputs(result_dir)
    ids = common_instance_ids(loaded)
    per_instance_rows = build_per_instance_rows(loaded, ids)
    summary_rows = build_summary_rows(loaded, ids)
    pair_count_rows, pair_percent_rows, win_percent = build_pairwise_tables(loaded, ids)

    write_main_tables(out_dir, summary_rows, pair_count_rows, pair_percent_rows, per_instance_rows)
    plot_average_cost(summary_rows, out_dir)
    plot_gap_to_lkh(summary_rows, out_dir)
    plot_gap_boxplot(per_instance_rows, out_dir)
    plot_pairwise_heatmap(win_percent, out_dir)
    plot_runtime(summary_rows, out_dir)
    plot_training_curves(out_dir)
    write_readme(out_dir, result_dir, len(ids))

    print(f"Saved summary: {out_dir.parent / 'summary.csv'}")
    print(f"Saved paper artifacts: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
