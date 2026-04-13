from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend — must be set before importing pyplot.
import matplotlib.pyplot as plt

from config import DATASET_NAME, DATA_DISTRIBUTION, STRATEGY_NAME


def plot_accuracy_chart(
    run_dir: Path,
    rounds: list[dict[str, float | int]],
) -> Path:
    """Plot accuracy vs. round and save to run_dir/accuracy.png."""
    out_path = run_dir / "accuracy.png"

    xs = [int(r["round"]) for r in rounds]
    ys = [float(r["accuracy"]) for r in rounds]
    n = len(xs)

    if ys and max(ys) <= 1.0:
        ys = [y * 100.0 for y in ys]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=120)
    fig.patch.set_facecolor("#E6E6E6")
    ax.set_facecolor("#E6E6E6")

    ax.plot(
        xs, ys,
        color="#0077B6",
        marker="o",
        markersize=5,
        markeredgecolor="#003049",
        linewidth=2.2,
        markevery=max(1, n // 25),
        label="Accuracy",
    )

    ax.set_title(f"{DATASET_NAME} · {DATA_DISTRIBUTION} · {STRATEGY_NAME}", fontsize=16, pad=10)
    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel("Accuracy %", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xlim(min(xs), max(xs))

    # Build clean x-tick positions.
    if n <= 25:
        xticks = xs
    else:
        step = max(1, n // 11)
        xticks = xs[::step]
        if xticks[-1] != xs[-1]:
            xticks.append(xs[-1])

    ax.set_xticks(xticks)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.45)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    return out_path
