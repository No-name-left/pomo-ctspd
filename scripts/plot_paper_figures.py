import argparse
import csv
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "test_results/thesis_main_synthetic_n100_g8_d1_with_lkh_20260427"
DEFAULT_DATASET_FILE = PROJECT_ROOT / "data/synthetic_tests/synthetic_n100_g8_d1_1000_seed20260423.pt"

MODEL_ORDER = [
    "LKH",
    "POMO baseline",
    "Full model",
    "scheduled bias",
    "w/o all bias",
    "w/o fusion gate",
    "w/o group emb.",
]

MODEL_KEYS = {
    "LKH": "lkh_low_first",
    "POMO baseline": "baseline",
    "Full model": "full_learnable",
    "scheduled bias": "scheduled_bias",
    "w/o all bias": "wo_all_bias",
    "w/o fusion gate": "wo_fusion_gate",
    "w/o group emb.": "wo_group_embedding",
}

KEY_TO_LABEL = {value: key for key, value in MODEL_KEYS.items()}

MODEL_COLORS = {
    "LKH": "#4D4D4D",
    "POMO baseline": "#999999",
    "Full model": "#0072B2",
    "scheduled bias": "#56B4E9",
    "w/o all bias": "#D55E00",
    "w/o fusion gate": "#009E73",
    "w/o group emb.": "#CC79A7",
}

GROUP_COLORS = [
    "#4D4D4D",
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#8C564B",
    "#7F7F7F",
]

NEURAL_ROUTE_MODELS = [
    "POMO baseline",
    "Full model",
    "scheduled bias",
    "w/o all bias",
    "w/o fusion gate",
    "w/o group emb.",
]

TRAINING_MAIN_MODELS = [
    "POMO baseline",
    "Full model",
    "scheduled bias",
    "w/o group emb.",
]

TRAINING_METRICS = {
    "POMO baseline": PROJECT_ROOT / "CSTPd_bsl/POMO/result/thesis_baseline_n100_g8_d1/training_metrics.csv",
    "Full model": PROJECT_ROOT
    / "CSTPd_cluster/POMO/result/25日_15点59分_cluster_n100_d1_new_full_learnable_bias/training_metrics.csv",
    "scheduled bias": PROJECT_ROOT / "CSTPd_cluster/POMO/result/ablation_scheduled_bias_n100_g8_d1/training_metrics.csv",
    "w/o all bias": PROJECT_ROOT / "CSTPd_cluster/POMO/result/26日_10点20分_cluster_n100_d1_wo_all_bias/training_metrics.csv",
    "w/o fusion gate": PROJECT_ROOT
    / "CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_fusion_gate_scheduled_bias_n100_g8_d1/training_metrics.csv",
    "w/o group emb.": PROJECT_ROOT
    / "CSTPd_cluster/POMO/result/legacy_struct_ablation_wo_group_embedding_scheduled_bias_n100_g8_d1/training_metrics.csv",
}

ROUTE_MODEL_CONFIGS = {
    "POMO baseline": {
        "model_type": "baseline",
        "model_variant": "baseline",
        "checkpoint_column": "checkpoint_path",
    },
    "Full model": {
        "model_type": "cluster",
        "model_variant": "learnable_bias",
        "checkpoint_column": "checkpoint_path",
    },
    "scheduled bias": {
        "model_type": "cluster",
        "model_variant": "scheduled_bias",
        "checkpoint_column": "checkpoint_path",
    },
    "w/o all bias": {
        "model_type": "cluster",
        "model_variant": "wo_all_bias",
        "checkpoint_column": "checkpoint_path",
    },
    "w/o fusion gate": {
        "model_type": "cluster",
        "model_variant": "wo_fusion_gate",
        "checkpoint_column": "checkpoint_path",
    },
    "w/o group emb.": {
        "model_type": "cluster",
        "model_variant": "wo_group_embedding",
        "checkpoint_column": "checkpoint_path",
    },
}

PLOT_STEMS = [
    "average_cost_with_lkh_bar",
    "gap_to_lkh_bar",
    "per_instance_gap_to_lkh_boxplot",
    "pairwise_win_heatmap",
    "time_per_instance_bar_log",
    "training_score_curves",
    "training_score_curves_appendix_all",
    "quality_time_tradeoff_scatter",
    "paired_gap_delta_to_baseline",
    "ablation_summary_delta",
    "feasibility_rate_bar",
    "route_case_study_panels",
    "route_case_group_sequence",
]


@dataclass
class FigureRecord:
    stem: str
    purpose: str
    placement: str
    source: str
    main_text: str
    caption_cn: str
    caption_en: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw thesis paper figures with a unified academic style."
    )
    parser.add_argument(
        "--result-dir",
        default=str(DEFAULT_RESULT_DIR),
        help="Main experiment result directory.",
    )
    parser.add_argument(
        "--dataset-file",
        default=str(DEFAULT_DATASET_FILE),
        help="Fixed synthetic .pt dataset used by the main experiment.",
    )
    parser.add_argument(
        "--output-subdir",
        default="paper_artifacts",
        help="Subdirectory under result-dir for figures.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used only for optional route re-inference.",
    )
    parser.add_argument(
        "--no-route",
        action="store_true",
        help="Skip route case-study inference and route figures.",
    )
    parser.add_argument(
        "--training-smooth-window",
        type=int,
        default=5,
        help="Rolling mean window for training curves. Use 1 for no smoothing.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E6E6E6", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")
    return [pdf_path, png_path]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        warnings.warn(f"missing CSV: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def to_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_label(row: dict[str, str]) -> str:
    key = row.get("model_key", "")
    if key in KEY_TO_LABEL:
        return KEY_TO_LABEL[key]
    raw = row.get("model_label", "")
    if raw == "Scheduled bias":
        return "scheduled bias"
    return raw


def ordered_summary_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(result_dir / "summary.csv")
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = normalize_label(row)
        row = dict(row)
        row["model_label"] = label
        if label in MODEL_ORDER:
            by_label[label] = row
    missing = [label for label in MODEL_ORDER if label not in by_label]
    if missing:
        raise FileNotFoundError(f"Missing models in summary.csv: {missing}")
    return [by_label[label] for label in MODEL_ORDER]


def load_per_instance(out_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(out_dir / "per_instance_costs.csv")
    if not rows:
        raise FileNotFoundError(f"Missing per-instance costs: {out_dir / 'per_instance_costs.csv'}")
    return rows


def short_number(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def add_bar_labels(ax: plt.Axes, bars: Any, values: list[float], fmt: str, dy: float = 3.0) -> None:
    for bar, value in zip(bars, values):
        if math.isnan(value):
            continue
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )


def model_colors(labels: list[str], alpha: float = 0.85) -> list[Any]:
    return [mpl.colors.to_rgba(MODEL_COLORS[label], alpha=alpha) for label in labels]


def plot_average_cost(summary_rows: list[dict[str, Any]], out_dir: Path) -> FigureRecord:
    labels = [row["model_label"] for row in summary_rows]
    means = [to_float(row["average_cost"]) for row in summary_rows]
    errors = [to_float(row.get("ci95_cost"), 0.0) for row in summary_rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=3,
        color=model_colors(labels),
        edgecolor="#333333",
        linewidth=0.6,
    )
    lower = max(0.0, math.floor((min(means) - 0.35) * 10) / 10)
    upper = math.ceil((max(means) + max(errors) + 0.25) * 10) / 10
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Average tour cost")
    ax.set_title("Average Tour Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    add_bar_labels(ax, bars, means, "{:.3f}")
    ax.text(
        0.01,
        0.02,
        "Y-axis truncated for readability",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#666666",
    )
    style_axes(ax)
    save_figure(fig, out_dir, "average_cost_with_lkh_bar")
    return FigureRecord(
        "average_cost_with_lkh_bar",
        "展示 LKH 与各神经模型的平均路径成本。",
        "主实验结果",
        "summary.csv",
        "是",
        "各方法在 1000 个 n=100, g=8, d=1 合成测试实例上的平均路径成本；误差线为 95% 置信区间，纵轴为便于比较进行了截断。",
        "Average tour cost over 1000 synthetic CTSP-d instances; error bars denote 95% confidence intervals.",
    )


def plot_gap_to_lkh(summary_rows: list[dict[str, Any]], out_dir: Path) -> FigureRecord:
    labels = [row["model_label"] for row in summary_rows]
    gaps = [to_float(row["average_gap_to_lkh_percent"]) for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    bars = ax.bar(
        x,
        gaps,
        color=model_colors(labels),
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_ylabel("Gap to LKH (%)")
    ax.set_title("Average Gap to LKH")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(min(-0.15, min(gaps) - 0.1), max(gaps) + 0.45)
    add_bar_labels(ax, bars, gaps, "{:.2f}%")
    style_axes(ax)
    save_figure(fig, out_dir, "gap_to_lkh_bar")
    return FigureRecord(
        "gap_to_lkh_bar",
        "展示各模型相对 LKH 的平均 gap。",
        "主实验结果",
        "summary.csv",
        "是",
        "各方法相对 LOW_FIRST-patched LKH 的平均 gap；Full model 是 gap 最小的神经模型。",
        "Average optimality gap relative to the LOW_FIRST-patched LKH reference.",
    )


def plot_gap_boxplot(per_instance_rows: list[dict[str, str]], out_dir: Path) -> FigureRecord:
    labels = [label for label in MODEL_ORDER if label != "LKH"]
    data = []
    for label in labels:
        key = MODEL_KEYS[label]
        data.append([to_float(row[f"{key}_gap_to_lkh_percent"]) for row in per_instance_rows])

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    flierprops = {
        "marker": "o",
        "markersize": 2.0,
        "markerfacecolor": "#777777",
        "markeredgecolor": "#777777",
        "alpha": 0.25,
    }
    meanprops = {
        "marker": "D",
        "markerfacecolor": "#FFFFFF",
        "markeredgecolor": "#333333",
        "markersize": 3.0,
    }
    box = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        flierprops=flierprops,
        meanprops=meanprops,
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(mpl.colors.to_rgba(MODEL_COLORS[label], alpha=0.45))
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    for item in box["medians"]:
        item.set_color("#111111")
        item.set_linewidth(1.1)
    for item in box["whiskers"] + box["caps"]:
        item.set_color("#333333")
        item.set_linewidth(0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    ax.set_ylabel("Per-instance gap to LKH (%)")
    ax.set_title("Per-instance Gap to LKH")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axes(ax)
    save_figure(fig, out_dir, "per_instance_gap_to_lkh_boxplot")
    return FigureRecord(
        "per_instance_gap_to_lkh_boxplot",
        "展示每个模型在逐实例层面的 gap 分布稳定性。",
        "主实验结果",
        "paper_artifacts/per_instance_costs.csv",
        "是",
        "各神经模型逐实例相对 LKH 的 gap 分布；箱体显示四分位数，菱形表示均值，离群点使用浅色小点显示。",
        "Distribution of per-instance gaps to LKH for neural models.",
    )


def plot_pairwise_heatmap(out_dir: Path) -> FigureRecord:
    rows = read_csv_rows(out_dir / "pairwise_win_percent.csv")
    if not rows:
        raise FileNotFoundError("pairwise_win_percent.csv is required for pairwise heatmap.")
    by_label = {normalize_label(row): row for row in rows}
    matrix = np.zeros((len(MODEL_ORDER), len(MODEL_ORDER)), dtype=float)
    for i, row_label in enumerate(MODEL_ORDER):
        row = by_label[row_label]
        for j, col_label in enumerate(MODEL_ORDER):
            matrix[i, j] = to_float(row[f"beats_{MODEL_KEYS[col_label]}_percent"])

    cmap = LinearSegmentedColormap.from_list(
        "paper_win_rate",
        ["#8C6D31", "#F7F7F7", "#2C7FB8"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    im = ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlabel("Compared method")
    ax.set_ylabel("Row method")
    ax.set_title("Pairwise Win Rate (%)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "#111111" if 32 <= matrix[i, j] <= 68 else "white"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8.2, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate (%)")
    style_axes(ax, grid_axis="")
    save_figure(fig, out_dir, "pairwise_win_heatmap")
    return FigureRecord(
        "pairwise_win_heatmap",
        "展示模型之间逐实例两两比较的胜率。",
        "消融分析",
        "paper_artifacts/pairwise_win_percent.csv",
        "是",
        "两两胜率矩阵，行方法相对列方法在同一测试实例上取得更低 cost 的比例；50% 为中点。",
        "Pairwise win-rate matrix; each cell is the percentage of instances where the row method beats the column method.",
    )


def plot_runtime(summary_rows: list[dict[str, Any]], out_dir: Path) -> FigureRecord:
    labels = [row["model_label"] for row in summary_rows]
    times = [to_float(row["time_per_instance_sec"]) for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    bars = ax.bar(
        x,
        times,
        color=model_colors(labels),
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Time per instance (s, log scale)")
    ax.set_title("Average Inference Time per Instance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for bar, value in zip(bars, times):
        ax.annotate(
            f"{value:.3f}s" if value < 1 else f"{value:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )
    full_time = times[labels.index("Full model")]
    lkh_time = times[labels.index("LKH")]
    if full_time > 0:
        ax.text(
            0.98,
            0.92,
            f"LKH / Full model: ×{lkh_time / full_time:.1f} time",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
        )
    style_axes(ax)
    ax.grid(axis="y", which="both", color="#E6E6E6", linewidth=0.6, alpha=0.6)
    save_figure(fig, out_dir, "time_per_instance_bar_log")
    return FigureRecord(
        "time_per_instance_bar_log",
        "展示 LKH 与神经模型的平均单实例推理时间。",
        "效率分析",
        "summary.csv",
        "是",
        "各方法平均单实例推理时间，纵轴为对数尺度；LKH 解质量更强但耗时约为 Full model 的百倍量级。",
        "Average inference time per instance on a log scale.",
    )


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_training_curves(out_dir: Path, window: int, appendix: bool) -> FigureRecord | None:
    labels = [label for label in MODEL_ORDER if label in TRAINING_METRICS]
    if not appendix:
        labels = TRAINING_MAIN_MODELS

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    plotted = []
    for label in labels:
        path = TRAINING_METRICS[label]
        rows = read_csv_rows(path)
        if not rows:
            continue
        epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch") and row.get("train_score")]
        scores = [float(row["train_score"]) for row in rows if row.get("epoch") and row.get("train_score")]
        if not epochs:
            continue
        smooth_scores = rolling_mean(scores, window)
        linewidth = 1.9 if label == "Full model" else 1.25
        linestyle = "--" if label == "POMO baseline" else "-"
        alpha = 1.0 if label in {"Full model", "POMO baseline"} else 0.88
        ax.plot(
            epochs,
            smooth_scores,
            label=label,
            color=MODEL_COLORS[label],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        plotted.append(label)
    if not plotted:
        plt.close(fig)
        warnings.warn("No training metrics found; skipped training curve.")
        return None
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training score")
    ax.set_title("Training Score Curves")
    ax.legend(loc="best", ncol=2 if appendix else 1)
    style_axes(ax)
    stem = "training_score_curves_appendix_all" if appendix else "training_score_curves"
    save_figure(fig, out_dir, stem)
    placement = "附录" if appendix else "训练过程分析"
    purpose = "展示全部神经模型训练 score 曲线。" if appendix else "展示主要模型训练 score 曲线。"
    main_text = "否，建议附录" if appendix else "可选"
    caption_cn = (
        f"全部神经模型训练 score 曲线，使用 rolling mean 平滑，窗口大小为 {window}。"
        if appendix
        else f"主要模型训练 score 曲线，使用 rolling mean 平滑，窗口大小为 {window}；为避免主图杂乱，仅展示 baseline、Full model、scheduled bias 和 w/o group emb."
    )
    return FigureRecord(
        stem,
        purpose,
        placement,
        "training_metrics.csv files under model result directories",
        main_text,
        caption_cn,
        "Training score curves with rolling-mean smoothing.",
    )


def plot_quality_time_tradeoff(summary_rows: list[dict[str, Any]], out_dir: Path) -> FigureRecord:
    labels = [row["model_label"] for row in summary_rows]
    times = [to_float(row["time_per_instance_sec"]) for row in summary_rows]
    gaps = [to_float(row["average_gap_to_lkh_percent"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    for label, x, y in zip(labels, times, gaps):
        size = 72 if label == "Full model" else 52
        ax.scatter(
            x,
            y,
            s=size,
            color=MODEL_COLORS[label],
            edgecolor="#333333",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        xytext = (6, 5)
        if label == "LKH":
            xytext = (-6, 8)
        elif label in {"POMO baseline", "scheduled bias"}:
            xytext = (6, -12)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext, fontsize=8.6)
    ax.set_xscale("log")
    ax.set_xlabel("Time per instance (s, log scale)")
    ax.set_ylabel("Gap to LKH (%)")
    ax.set_title("Quality-Time Tradeoff")
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    style_axes(ax)
    save_figure(fig, out_dir, "quality_time_tradeoff_scatter")
    return FigureRecord(
        "quality_time_tradeoff_scatter",
        "展示解质量与推理时间之间的折中关系。",
        "效率分析",
        "summary.csv",
        "是",
        "解质量与推理时间折中图；LKH 的 gap 为 0 但耗时显著更高，Full model 在神经模型中取得较低 gap 与较高速度的折中。",
        "Quality-time tradeoff: average gap to LKH versus inference time per instance.",
    )


def plot_paired_delta_to_baseline(per_instance_rows: list[dict[str, str]], out_dir: Path) -> FigureRecord:
    labels = [
        "Full model",
        "scheduled bias",
        "w/o all bias",
        "w/o fusion gate",
        "w/o group emb.",
    ]
    baseline_key = MODEL_KEYS["POMO baseline"]
    data = []
    for label in labels:
        key = MODEL_KEYS[label]
        values = []
        for row in per_instance_rows:
            model_gap = to_float(row[f"{key}_gap_to_lkh_percent"])
            base_gap = to_float(row[f"{baseline_key}_gap_to_lkh_percent"])
            values.append(model_gap - base_gap)
        data.append(values)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    box = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.0,
            "markerfacecolor": "#777777",
            "markeredgecolor": "#777777",
            "alpha": 0.25,
        },
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(mpl.colors.to_rgba(MODEL_COLORS[label], alpha=0.45))
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    for item in box["medians"]:
        item.set_color("#111111")
        item.set_linewidth(1.1)
    for item in box["whiskers"] + box["caps"]:
        item.set_color("#333333")
        item.set_linewidth(0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Delta gap vs baseline (pp; lower is better)")
    ax.set_title("Paired Gap Delta to Baseline")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axes(ax)
    save_figure(fig, out_dir, "paired_gap_delta_to_baseline")
    return FigureRecord(
        "paired_gap_delta_to_baseline",
        "展示每个模型相对 baseline 的逐实例 gap 改变量。",
        "消融分析",
        "paper_artifacts/per_instance_costs.csv",
        "是",
        "逐实例相对 baseline 的 gap 差值分布；数值小于 0 表示该模型在对应实例上优于 baseline。",
        "Paired per-instance gap delta against the POMO baseline; negative values indicate improvements.",
    )


def plot_ablation_delta(summary_rows: list[dict[str, Any]], out_dir: Path) -> FigureRecord:
    full_gap = next(to_float(row["average_gap_to_lkh_percent"]) for row in summary_rows if row["model_label"] == "Full model")
    labels = ["scheduled bias", "w/o all bias", "w/o fusion gate", "w/o group emb."]
    by_label = {row["model_label"]: row for row in summary_rows}
    deltas = [to_float(by_label[label]["average_gap_to_lkh_percent"]) - full_gap for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    bars = ax.bar(
        x,
        deltas,
        color=model_colors(labels),
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Delta gap vs Full model (pp)")
    ax.set_title("Ablation Delta vs Full Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    add_bar_labels(ax, bars, deltas, "{:+.2f}", dy=3)
    ax.text(
        0.02,
        0.92,
        "Positive = worse than Full model",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#666666",
    )
    style_axes(ax)
    save_figure(fig, out_dir, "ablation_summary_delta")
    return FigureRecord(
        "ablation_summary_delta",
        "直接展示各消融模型相对 Full model 的平均 gap 变化。",
        "消融分析",
        "summary.csv",
        "是",
        "各消融模型相对 Full model 的平均 gap 增量；正值表示去掉或替换对应设计后性能变差。",
        "Average gap delta of ablation variants relative to the Full model.",
    )


def plot_feasibility(summary_rows: list[dict[str, Any]], out_dir: Path) -> FigureRecord:
    labels = [row["model_label"] for row in summary_rows]
    rates = [100.0 * to_float(row["feasible_rate"]) for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bars = ax.bar(
        x,
        rates,
        color=model_colors(labels),
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Feasible rate (%)")
    ax.set_title("Feasibility Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    add_bar_labels(ax, bars, rates, "{:.1f}%")
    style_axes(ax)
    save_figure(fig, out_dir, "feasibility_rate_bar")
    return FigureRecord(
        "feasibility_rate_bar",
        "展示各方法生成解的 CTSP-d 可行率。",
        "附录",
        "summary.csv and evaluation test_instances.csv",
        "否，建议附录",
        "各方法在主实验测试集上的可行率；当前所有方法均为 100%，说明主结果差异主要来自路径成本而非约束失败。",
        "Feasibility rate on the main synthetic test set.",
    )


def import_evaluate_module() -> Any:
    path = PROJECT_ROOT / "scripts/evaluate_ctspd.py"
    spec = importlib.util.spec_from_file_location("evaluate_ctspd_for_figures", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_tour_file(path: Path) -> list[int]:
    tour = []
    reading = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "TOUR_SECTION":
            reading = True
            continue
        if not reading:
            continue
        if line in {"-1", "EOF"}:
            break
        if line:
            tour.append(int(line) - 1)
    return tour


def tour_cost(tour: list[int], coords: np.ndarray) -> float:
    arr = np.asarray(tour, dtype=np.int64)
    xy = coords[arr]
    nxt = coords[np.roll(arr, -1)]
    return float(np.linalg.norm(xy - nxt, axis=1).sum())


def calc_feasibility_np(tour: list[int], priorities: np.ndarray, relaxation_d: int) -> tuple[int, bool]:
    n_nodes = int(priorities.shape[0])
    unvisited = set(range(n_nodes))
    violations = 0
    if len(tour) != n_nodes or len(set(tour)) != n_nodes:
        violations += abs(len(tour) - n_nodes) + (len(tour) - len(set(tour)))
    for node in tour:
        if node not in unvisited:
            violations += 1
            continue
        current_min = min(float(priorities[idx]) for idx in unvisited)
        node_priority = float(priorities[node])
        if node_priority < current_min or node_priority > current_min + relaxation_d:
            violations += 1
        unvisited.remove(node)
    violations += len(unvisited)
    return violations, violations == 0


def choose_case_instance(per_instance_rows: list[dict[str, str]]) -> dict[str, Any]:
    candidates = []
    for row in per_instance_rows:
        base = to_float(row["baseline_cost"])
        full = to_float(row["full_learnable_cost"])
        improvement = base - full
        if improvement > 0:
            candidates.append((row["instance_id"], improvement))
    if not candidates:
        row = per_instance_rows[len(per_instance_rows) // 2]
        return {"instance_id": row["instance_id"], "selection_rule": "median row fallback"}
    values = np.asarray([item[1] for item in candidates], dtype=float)
    low, high = np.quantile(values, [0.10, 0.90])
    filtered = [(inst, imp) for inst, imp in candidates if low <= imp <= high]
    if not filtered:
        filtered = candidates
    target = float(np.quantile([item[1] for item in filtered], 0.60))
    instance_id, improvement = min(filtered, key=lambda item: (abs(item[1] - target), item[0]))
    return {
        "instance_id": instance_id,
        "selection_rule": "positive baseline-full improvement nearest 60th percentile after excluding 10% tails",
        "baseline_minus_full_cost": improvement,
        "target_improvement": target,
    }


def load_dataset_instance(dataset_file: Path, instance_id: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(dataset_file, map_location="cpu")
    problems = payload["problems"]
    metadata = payload.get("metadata", {})
    index = int(instance_id.split("_")[-1])
    if index < 0 or index >= int(problems.size(0)):
        raise IndexError(f"{instance_id} is outside {dataset_file}")
    problem = problems[index : index + 1].clone()
    coords = problem[0, :, :2].detach().cpu().numpy().astype(float)
    priorities = problem[0, :, 2].detach().cpu().numpy().astype(int)
    return problem, coords, priorities, metadata


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(device_arg)


def infer_neural_routes(
    summary_rows: list[dict[str, Any]],
    problem: torch.Tensor,
    metadata: dict[str, Any],
    device_arg: str,
) -> dict[str, dict[str, Any]]:
    evaluate = import_evaluate_module()
    device = resolve_device(device_arg)
    torch.manual_seed(20260423)
    evaluate.set_default_device(device)
    summary_by_label = {row["model_label"]: row for row in summary_rows}
    routes: dict[str, dict[str, Any]] = {}
    for label in NEURAL_ROUTE_MODELS:
        config = ROUTE_MODEL_CONFIGS[label]
        checkpoint = resolve_path(summary_by_label[label][config["checkpoint_column"]])
        if not checkpoint.exists():
            warnings.warn(f"missing checkpoint for {label}: {checkpoint}")
            continue
        model, env_cls, _params, _checkpoint = evaluate.load_model(
            config["model_type"],
            config["model_variant"],
            checkpoint,
            device,
        )
        result = evaluate.run_best_instance(
            model,
            env_cls,
            problem,
            raw_dist=None,
            relaxation_d=int(metadata.get("relaxation_d", 1)),
            num_groups=int(metadata.get("num_groups", int(problem[0, :, 2].max().item()) + 1)),
            device=device,
            augmentation_factor=8,
            feature_modes=["anchor"],
            sampling_runs=0,
            sampling_temperatures=[1.0],
            sampling_top_ks=[0],
        )
        routes[label] = {
            "tour": [int(node) for node in result["tour"]],
            "derived_cost": float(result["cost"]),
            "feasible": bool(result["feasible"]),
            "violation_count": int(result["violation_count"]),
            "best_batch_idx": int(result["best_batch_idx"]),
            "best_pomo_idx": int(result["best_pomo_idx"]),
        }
        print(f"inferred route for {label}: cost={float(result['cost']):.6f}")
    return routes


def load_lkh_route(result_dir: Path, instance_id: str) -> dict[str, Any] | None:
    rows = read_csv_rows(result_dir / "evaluations/lkh_low_first/lkh_instances.csv")
    for row in rows:
        if row.get("instance_id") == instance_id:
            path_text = row.get("normalized_tour_file")
            if not path_text:
                return None
            path = Path(path_text)
            if not path.exists():
                warnings.warn(f"missing LKH normalized tour: {path}")
                return None
            return {
                "tour": parse_tour_file(path),
                "reported_cost": to_float(row.get("lkh_cost")),
                "feasible": str(row.get("is_feasible", "")).lower() == "true",
                "violation_count": int(float(row.get("violation_count", "0"))),
                "path": str(path),
            }
    return None


def enrich_route_metadata(
    routes: dict[str, dict[str, Any]],
    per_instance_by_id: dict[str, dict[str, str]],
    instance_id: str,
    coords: np.ndarray,
    priorities: np.ndarray,
    relaxation_d: int,
) -> None:
    row = per_instance_by_id[instance_id]
    for label, data in routes.items():
        key = MODEL_KEYS[label]
        data["reported_cost"] = to_float(row[f"{key}_cost"])
        data["reported_gap_to_lkh_percent"] = to_float(row[f"{key}_gap_to_lkh_percent"])
        data["route_cost_from_coords"] = tour_cost(data["tour"], coords)
        violations, feasible = calc_feasibility_np(data["tour"], priorities, relaxation_d)
        data["route_violation_count"] = violations
        data["route_feasible"] = feasible


def plot_route_panels(
    routes: dict[str, dict[str, Any]],
    coords: np.ndarray,
    priorities: np.ndarray,
    out_dir: Path,
    instance_id: str,
) -> FigureRecord:
    labels = [label for label in MODEL_ORDER if label in routes]
    n_cols = 4
    n_rows = math.ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.2, 3.4 * n_rows), squeeze=False)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    pad = 0.035
    group_values = sorted(int(value) for value in np.unique(priorities))
    group_to_color = {group: GROUP_COLORS[group % len(GROUP_COLORS)] for group in group_values}

    for idx, label in enumerate(labels):
        ax = axes[idx // n_cols][idx % n_cols]
        tour = routes[label]["tour"]
        route_xy = coords[tour + [tour[0]]]
        ax.plot(
            route_xy[:, 0],
            route_xy[:, 1],
            color=MODEL_COLORS[label],
            linewidth=0.8,
            alpha=0.72,
            zorder=2,
        )
        for group in group_values:
            mask = priorities == group
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=13,
                color=group_to_color[group],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.92,
                zorder=3,
            )
        start = tour[0]
        ax.scatter(
            [coords[start, 0]],
            [coords[start, 1]],
            marker="*",
            s=80,
            color="#FFFFFF",
            edgecolor="#111111",
            linewidth=0.8,
            zorder=4,
        )
        cost = routes[label].get("reported_cost", routes[label].get("derived_cost", math.nan))
        gap = routes[label].get("reported_gap_to_lkh_percent", math.nan)
        ax.set_title(f"{label}\ncost={cost:.3f}, gap={gap:.2f}%", fontsize=10.5)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        style_axes(ax, grid_axis="")
    for idx in range(len(labels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label=f"Group {group}", markerfacecolor=group_to_color[group], markersize=5)
        for group in group_values
    ]
    legend_handles.append(
        Line2D([0], [0], marker="*", color="none", label="Start", markerfacecolor="white", markeredgecolor="#111111", markersize=8)
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 9),
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle(f"Route Case Study: {instance_id}", fontsize=13, y=0.995)
    fig.subplots_adjust(bottom=0.10 if n_rows == 1 else 0.07, top=0.88 if n_rows == 1 else 0.92)
    save_figure(fig, out_dir, "route_case_study_panels")
    return FigureRecord(
        "route_case_study_panels",
        "展示同一代表性实例上不同模型路线形态的差异。",
        "附录或案例分析",
        "Fixed synthetic dataset, per_instance_costs.csv, LKH normalized tour, and checkpoint re-inference for neural routes",
        "可选，正文空间允许时可放",
        f"代表性实例 {instance_id} 的路线案例图。节点颜色表示 priority group，线颜色表示模型，星号为路线起点。",
        "Route case study for one representative instance; node colors indicate priority groups.",
    )


def plot_group_sequence(
    routes: dict[str, dict[str, Any]],
    priorities: np.ndarray,
    out_dir: Path,
    instance_id: str,
) -> FigureRecord:
    labels = [label for label in MODEL_ORDER if label in routes]
    sequences = np.asarray([[priorities[node] for node in routes[label]["tour"]] for label in labels], dtype=int)
    unique_groups = sorted(int(value) for value in np.unique(priorities))
    cmap = ListedColormap([GROUP_COLORS[group % len(GROUP_COLORS)] for group in unique_groups])
    bounds = [group - 0.5 for group in unique_groups] + [unique_groups[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(labels) + 2.0))
    im = ax.imshow(sequences, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Step / order")
    ax.set_title(f"Priority Group Visit Sequence: {instance_id}")
    n_steps = sequences.shape[1]
    ticks = sorted(set([0, 24, 49, 74, n_steps - 1]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick + 1) for tick in ticks])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, ticks=unique_groups)
    cbar.set_label("Priority group")
    style_axes(ax, grid_axis="")
    save_figure(fig, out_dir, "route_case_group_sequence")
    return FigureRecord(
        "route_case_group_sequence",
        "展示同一实例中不同模型访问 priority group 的顺序差异。",
        "附录或案例分析",
        "Fixed synthetic dataset, LKH normalized tour, and checkpoint re-inference for neural routes",
        "可选，正文空间允许时可放",
        f"代表性实例 {instance_id} 的 priority group 访问序列；每一行是一个模型，每一列是访问步序。",
        "Priority-group visit sequence for the route case study.",
    )


def build_route_case_study(
    result_dir: Path,
    out_dir: Path,
    dataset_file: Path,
    summary_rows: list[dict[str, Any]],
    per_instance_rows: list[dict[str, str]],
    device_arg: str,
) -> tuple[list[FigureRecord], list[str]]:
    missing_notes = []
    try:
        selection = choose_case_instance(per_instance_rows)
        instance_id = selection["instance_id"]
        problem, coords, priorities, metadata = load_dataset_instance(dataset_file, instance_id)
        routes = infer_neural_routes(summary_rows, problem, metadata, device_arg)
        lkh_route = load_lkh_route(result_dir, instance_id)
        if lkh_route:
            routes = {"LKH": lkh_route, **routes}
        per_instance_by_id = {row["instance_id"]: row for row in per_instance_rows}
        enrich_route_metadata(
            routes,
            per_instance_by_id,
            instance_id,
            coords,
            priorities,
            int(metadata.get("relaxation_d", 1)),
        )
        if not all(label in routes for label in NEURAL_ROUTE_MODELS):
            missing = [label for label in NEURAL_ROUTE_MODELS if label not in routes]
            raise RuntimeError(f"Missing neural routes for {missing}")
        records = [
            plot_route_panels(routes, coords, priorities, out_dir, instance_id),
            plot_group_sequence(routes, priorities, out_dir, instance_id),
        ]
        route_summary = {
            "instance_id": instance_id,
            "selection": selection,
            "dataset_file": str(dataset_file),
            "route_source": "Neural routes were deterministically re-inferred from checkpoints with 8-fold augmentation; LKH route came from normalized_tour_file.",
            "models": {
                label: {
                    key: value
                    for key, value in data.items()
                    if key not in {"tour"}
                }
                for label, data in routes.items()
            },
        }
        (out_dir / "route_case_study_summary.json").write_text(
            json.dumps(route_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved {out_dir / 'route_case_study_summary.json'}")
        return records, missing_notes
    except Exception as exc:
        note = f"route_case_study_panels / route_case_group_sequence 未生成：{exc}"
        warnings.warn(note)
        missing_notes.append(note)
        return [], missing_notes


def write_manifest(out_dir: Path, records: list[FigureRecord], missing_notes: list[str]) -> None:
    rows = []
    for record in records:
        for suffix in ["pdf", "png"]:
            path = out_dir / f"{record.stem}.{suffix}"
            if path.exists():
                rows.append(
                    {
                        "file": path.name,
                        "stem": record.stem,
                        "format": suffix,
                        "purpose": record.purpose,
                        "placement": record.placement,
                        "source": record.source,
                        "main_text": record.main_text,
                    }
                )
    write_csv(
        out_dir / "FIGURE_MANIFEST.csv",
        rows,
        ["file", "stem", "format", "purpose", "placement", "source", "main_text"],
    )
    if missing_notes:
        (out_dir / "MISSING_FIGURES.md").write_text(
            "# Missing or Skipped Figures\n\n" + "\n".join(f"- {note}" for note in missing_notes) + "\n",
            encoding="utf-8",
        )
        print(f"saved {out_dir / 'MISSING_FIGURES.md'}")
    else:
        missing_path = out_dir / "MISSING_FIGURES.md"
        if missing_path.exists():
            missing_path.unlink()
            print(f"removed {missing_path}")


def write_figure_handoff(
    out_dir: Path,
    result_dir: Path,
    records: list[FigureRecord],
    missing_notes: list[str],
    smooth_window: int,
) -> None:
    lines = [
        "# Figure Handoff",
        "",
        f"Updated: 2026-04-27",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "All figures are regenerated by `scripts/plot_paper_figures.py` with a unified paper style: serif font, white background, muted colorblind-friendly model palette, dark gray axes, shallow horizontal grids, vector PDF output, and 300 dpi PNG preview output.",
        "",
        f"Training curves use rolling-mean smoothing with window size `{smooth_window}`. This smoothing is applied only to the plotted curves, not to the original training logs.",
        "",
        "Model colors:",
        "",
    ]
    for label in MODEL_ORDER:
        lines.append(f"- `{label}`: `{MODEL_COLORS[label]}`")
    lines.extend(["", "## Generated Figures", ""])
    for record in records:
        pdf = out_dir / f"{record.stem}.pdf"
        png = out_dir / f"{record.stem}.png"
        lines.extend(
            [
                f"### `{record.stem}.pdf/png`",
                "",
                f"- Purpose: {record.purpose}",
                f"- Recommended placement: {record.placement}",
                f"- Data source: {record.source}",
                f"- Suitable for main text: {record.main_text}",
                f"- Files: `{pdf.name}`, `{png.name}`",
                f"- 中文图注：{record.caption_cn}",
                f"- English caption: {record.caption_en}",
                "",
            ]
        )
    lines.extend(["## Skipped or Not Generated", ""])
    if missing_notes:
        lines.extend(f"- {note}" for note in missing_notes)
    lines.extend(
        [
            "- `generalization_or_sensitivity_curves` not generated: the current committed main experiment has one controlled setting only (`n=100`, `num_groups=8`, `d=1`) and does not contain comparable LKH-backed sweeps over `n`, `d`, or `num_groups`.",
            "- `violation_count_boxplot` not generated separately: all methods have feasible rate 100% and violation count 0 in the available main-result summaries; `feasibility_rate_bar` records the constraint result more compactly.",
            "",
            "## Notes",
            "",
            "- Existing six figure names were reused and overwritten with the new style, so old duplicate versions are not kept.",
            "- Route case-study neural tours, when present, are derived by deterministic checkpoint re-inference on a single selected synthetic instance with 8-fold augmentation. This does not modify the original experiment outputs.",
            "- PDF files should be used for thesis insertion; PNG files are previews.",
            "",
        ]
    )
    path = out_dir / "FIGURE_HANDOFF.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {path}")


def write_readme(out_dir: Path, records: list[FigureRecord]) -> None:
    lines = [
        "# Paper Artifacts",
        "",
        "This folder contains thesis-ready figures, tables, and handoff notes for the synthetic CTSP-d main experiment with LKH.",
        "",
        "Use the `.pdf` files in the thesis whenever possible. The `.png` files are 300 dpi previews.",
        "",
        "Key files:",
        "",
        "- `FIGURE_HANDOFF.md`: detailed description, recommended placement, and captions for each figure.",
        "- `FIGURE_MANIFEST.csv`: machine-readable figure inventory.",
        "- `main_results_table.csv` / `.md`: main result table.",
        "- `ablation_table.csv` / `.md`: ablation table.",
        "- `per_instance_costs.csv`: aligned per-instance costs and gaps.",
        "- `pairwise_win_counts.csv` / `pairwise_win_percent.csv`: pairwise comparison tables.",
        "",
        "Generated figures:",
        "",
    ]
    for record in records:
        lines.append(f"- `{record.stem}.pdf/png`: {record.purpose}")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out_dir / 'README.md'}")


def remove_stale_plot_files(out_dir: Path, generated_records: list[FigureRecord]) -> list[str]:
    expected = {f"{record.stem}.{suffix}" for record in generated_records for suffix in ["pdf", "png"]}
    expected.update(
        {
            "README.md",
            "FIGURE_HANDOFF.md",
            "FIGURE_MANIFEST.csv",
            "route_case_study_summary.json",
            "MISSING_FIGURES.md",
            "main_results_table.csv",
            "main_results_table.md",
            "ablation_table.csv",
            "ablation_table.md",
            "per_instance_costs.csv",
            "pairwise_win_counts.csv",
            "pairwise_win_percent.csv",
        }
    )
    removed = []
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"} and path.name not in expected:
            path.unlink()
            removed.append(path.name)
            print(f"removed stale figure {path}")
    return removed


def main() -> None:
    args = parse_args()
    result_dir = resolve_path(args.result_dir)
    dataset_file = resolve_path(args.dataset_file)
    out_dir = result_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()
    summary_rows = ordered_summary_rows(result_dir)
    per_instance_rows = load_per_instance(out_dir)

    records: list[FigureRecord] = []
    missing_notes: list[str] = []

    records.append(plot_average_cost(summary_rows, out_dir))
    records.append(plot_gap_to_lkh(summary_rows, out_dir))
    records.append(plot_gap_boxplot(per_instance_rows, out_dir))
    records.append(plot_pairwise_heatmap(out_dir))
    records.append(plot_runtime(summary_rows, out_dir))
    main_training = plot_training_curves(out_dir, args.training_smooth_window, appendix=False)
    if main_training:
        records.append(main_training)
    appendix_training = plot_training_curves(out_dir, args.training_smooth_window, appendix=True)
    if appendix_training:
        records.append(appendix_training)
    records.append(plot_quality_time_tradeoff(summary_rows, out_dir))
    records.append(plot_paired_delta_to_baseline(per_instance_rows, out_dir))
    records.append(plot_ablation_delta(summary_rows, out_dir))
    records.append(plot_feasibility(summary_rows, out_dir))

    if args.no_route:
        missing_notes.append("route_case_study_panels / route_case_group_sequence skipped by --no-route.")
    else:
        route_records, route_missing = build_route_case_study(
            result_dir,
            out_dir,
            dataset_file,
            summary_rows,
            per_instance_rows,
            args.device,
        )
        records.extend(route_records)
        missing_notes.extend(route_missing)

    removed = remove_stale_plot_files(out_dir, records)
    if removed:
        missing_notes.append(f"Removed stale duplicate figure files: {', '.join(sorted(removed))}.")

    write_manifest(out_dir, records, missing_notes)
    write_figure_handoff(out_dir, result_dir, records, missing_notes, args.training_smooth_window)
    write_readme(out_dir, records)
    print(f"Generated {len(records)} figure groups in {out_dir}")


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
