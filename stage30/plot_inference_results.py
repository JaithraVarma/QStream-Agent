"""Render the final graph comparisons from the saved experiment receipts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent / "final-checks"
FP16 = "#52627B"
INT4 = "#087F72"
K32 = "#B44E30"


def load(name):
    result = json.loads((OUT / name).read_text())
    if result["status"] != "PASS":
        raise ValueError(f"{name} did not pass")
    return result


def main():
    precision = load("precision.json")
    factorial = load("factorial.json")
    cases = precision["timing"]
    lengths = [case["prompt_tokens"] for case in cases]
    labels = ("Stock FP16", "Kimi INT4")
    variants = ("fp16", "kimi_int4")
    colors = (FP16, INT4)
    fig, (latency, memory, stages) = plt.subplots(1, 3, figsize=(13.2, 4.4))

    for offset, (variant, label, color) in enumerate(zip(variants, labels, colors)):
        xs = [index + (offset - .5) * .30 for index in range(len(cases))]
        ys = [case["medians_ms_per_step"][variant]["wall_ms"] for case in cases]
        latency.scatter(xs, ys, color=color, s=48, label=label, zorder=3)
        for x, value in zip(xs, ys):
            latency.annotate(f"{value:.3f}", (x, value), xytext=(0, 8),
                             textcoords="offset points", ha="center", color=color, fontsize=9)
        peaks = [precision["memory"][variant]["cases"][str(length)]["peak_allocated_bytes"] / 2**20
                 for length in lengths]
        bars = memory.bar(xs, peaks, color=color, width=.27)
        memory.bar_label(bars, labels=[f"{value:,.0f}" for value in peaks], padding=4, fontsize=9)

    for width, color in ((32, K32), (64, INT4)):
        ys = [factorial["medians_ms_per_step"][f"k{width}_s{stage}"]["wall_ms"] for stage in (1, 2)]
        stages.plot((1, 2), ys, color=color, marker="o", label=f"K tile {width}", linewidth=2)
    stages.set_xticks((1, 2), ("1 stage", "2 stages"))
    stages.set_xlim(.8, 2.2)
    stages.set_ylim(7.06, 7.50)
    stages.set_ylabel("Wall time (ms/token)")
    stages.legend(frameon=False, loc="upper left", fontsize=9)
    stages.set_title("C   Stage count at 128 tokens", loc="left", fontweight="bold")

    for ax, title, ylabel in ((latency, "A   Decode latency", "Wall time (ms/token)"),
                              (memory, "B   Peak allocated memory", "PyTorch peak (MiB)")):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(lengths)), lengths)
        ax.set_xlabel("Prompt tokens")
        ax.set_xlim(-.45, len(lengths) - .55)
    latency.set_ylim(6.95, 8.78)
    memory.set_ylim(0, 1430)
    for ax in (latency, memory, stages):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("QStream-Agent on Qwen3-0.6B", x=.055, ha="left", fontsize=16, fontweight="bold")
    fig.legend(*latency.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(.96, .975),
               frameon=False, ncol=2)
    fig.text(.055, .028, "RTX 3060 Laptop GPU • batch 1 • 16 cached decode steps per trial. "
             "Panel B measures each model separately after graph capture.", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.065, right=.97, top=.82, bottom=.22, wspace=.43)
    fig.savefig(OUT / "inference-results.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
