from pathlib import Path

import matplotlib

# Use a non-GUI backend to avoid Tkinter/thread issues in Flower worker threads.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATASET_NAME, DATA_DISTRIBUTION, STRATEGY_NAME


def draw_acc_chart(experiment_dir: Path, records: list[dict[str, float | int]]) -> Path:
    """Draw and save accuracy-by-round chart."""
    chart_path = experiment_dir / "acc_round.png"

    rounds = [int(item["round"]) for item in records]
    accs = [float(item["accuracy"]) for item in records]
    n_points = len(rounds)

    if accs and max(accs) <= 1.0:
        accs = [value * 100.0 for value in accs]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=120)
    fig.patch.set_facecolor("#E6E6E6")
    ax.set_facecolor("#E6E6E6")

    marker_step = max(1, n_points // 25)

    ax.plot(
        rounds,
        accs,
        color="#0077B6",
        marker="o",
        markersize=5,
        markeredgecolor="#003049",
        linewidth=2.2,
        markevery=marker_step,
        label="Model Accuracy",
    )

    run_title = f"{DATASET_NAME}-{DATA_DISTRIBUTION}-{STRATEGY_NAME}"
    ax.set_title(run_title, fontsize=18, pad=10)
    ax.set_xlabel("Training iteration", fontsize=16)
    ax.set_ylabel("Accuracy %", fontsize=16)

    if n_points <= 25:
        xticks = rounds
    else:
        tick_count = 12
        step = max(1, n_points // (tick_count - 1))
        xticks = [rounds[idx] for idx in range(0, n_points, step)]
        if xticks[-1] != rounds[-1]:
            if rounds[-1] - xticks[-1] <= max(1, step // 2):
                xticks[-1] = rounds[-1]
            else:
                xticks.append(rounds[-1])

    ax.set_xticks(xticks)
    ax.set_xlim(min(rounds), max(rounds))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.45)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.2)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_ylim(0, 100)

    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=240)
    plt.close(fig)

    return chart_path
