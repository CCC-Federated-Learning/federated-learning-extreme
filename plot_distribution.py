import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime

from config import (
    BATCH_SIZE,
    CHART_DIR,
    CHART_FIGSIZE,
    CHART_NAME,
    CHART_SHOW,
    DATA_DISTRIBUTION,
    DATA_SEED,
    DATASET_NAME,
    DIRICHLET_ALPHA,
    NUM_CLIENTS,
    STRATEGY_NAME,
    TIMESTAMP_FORMAT,
)
from app.model import load_client_data

_DIGIT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_distribution(
    dataset_name=DATASET_NAME,
    num_clients=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
    distribution=DATA_DISTRIBUTION,
    dirichlet_alpha=DIRICHLET_ALPHA,
    seed=DATA_SEED,
    save_dir=CHART_DIR,
    save_name=CHART_NAME,
    show=CHART_SHOW,
    add_timestamp=True,
    strategy_name=STRATEGY_NAME,
) -> None:
    """Plot each client's label distribution as a stacked bar chart."""
    effective_seed = (
        int(datetime.now().timestamp() * 1_000_000) % (2 ** 32)
        if seed is None
        else seed
    )

    counts = torch.zeros((num_clients, 10), dtype=torch.int64)
    for cid in range(num_clients):
        trainloader, _ = load_client_data(
            cid, num_clients, batch_size,
            distribution=distribution, dirichlet_alpha=dirichlet_alpha, seed=effective_seed,
        )
        subset = trainloader.dataset
        labels = subset.dataset.targets[subset.indices]
        unique, cnts = torch.unique(labels, return_counts=True)
        counts[cid, unique.long()] = cnts.long()

    client_labels = [f"C{i}" for i in range(num_clients)]
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=100)
    fig.patch.set_facecolor("#E6E6E6")
    ax.set_facecolor("#E6E6E6")

    bottom = torch.zeros(num_clients, dtype=torch.int64)
    for digit in range(10):
        values = counts[:, digit]
        ax.bar(
            client_labels,
            values.tolist(),
            bottom=bottom.tolist(),
            color=_DIGIT_COLORS[digit],
            linewidth=0,
            label=str(digit),
        )
        bottom += values

    ax.set_title(f"{dataset_name} · {distribution} · {strategy_name}")
    ax.set_xlabel("Client")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend(title="Digit", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(save_name).stem
    suffix = Path(save_name).suffix or ".png"
    if add_timestamp:
        ts = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename = f"{stem}_{ts}{suffix}"
    else:
        filename = f"{stem}{suffix}"

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Distribution chart saved: {out_path.resolve()}  (seed={effective_seed})")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    plot_distribution()
