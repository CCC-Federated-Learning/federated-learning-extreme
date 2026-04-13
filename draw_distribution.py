import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime

from config import (
    BATCH_SIZE,
    DATASET_NAME,
    DATA_DISTRIBUTION,
    DIRICHLET_ALPHA,
    DATA_SEED,
    DRAW_FIGSIZE,
    DRAW_SAVE_DIR,
    DRAW_SAVE_NAME,
    DRAW_SHOW_PLOT,
    NUM_PARTITIONS,
    STRATEGY_NAME,
    TIMESTAMP_FORMAT,
)
from app.task import load_data


def plot_client_distribution(
    dataset_name=DATASET_NAME,
    num_partitions=NUM_PARTITIONS,
    batch_size=BATCH_SIZE,
    distribution=DATA_DISTRIBUTION,
    dirichlet_alpha=DIRICHLET_ALPHA,
    seed=DATA_SEED,
    save_dir=DRAW_SAVE_DIR,
    save_name=DRAW_SAVE_NAME,
    show_plot=DRAW_SHOW_PLOT,
    add_timestamp=True,
    strategy_name=STRATEGY_NAME,
):
    """Plot each client's label distribution as a stacked bar chart."""
    effective_seed = (
        int(datetime.now().timestamp() * 1_000_000) % (2**32)
        if seed is None
        else seed
    )

    client_labels = [f"C{i}" for i in range(num_partitions)]
    counts_matrix = torch.zeros((num_partitions, 10), dtype=torch.int64)

    for partition_id in range(num_partitions):
        trainloader, _ = load_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            distribution=distribution,
            dirichlet_alpha=dirichlet_alpha,
            seed=effective_seed,
        )

        # For Subset datasets, labels are available through dataset.indices.
        subset = trainloader.dataset
        labels = subset.dataset.targets[subset.indices]
        unique_labels, counts = torch.unique(labels, return_counts=True)
        counts_matrix[partition_id, unique_labels.long()] = counts.long()

    # Discrete and high-contrast colors (non-gradient), one for each digit.
    digit_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig, ax = plt.subplots(figsize=DRAW_FIGSIZE, dpi=100)
    fig.patch.set_facecolor("#E6E6E6")
    ax.set_facecolor("#E6E6E6")

    bottom = torch.zeros(num_partitions, dtype=torch.int64)
    for digit in range(10):
        values = counts_matrix[:, digit]
        ax.bar(
            client_labels,
            values.tolist(),
            bottom=bottom.tolist(),
            color=digit_colors[digit],
            linewidth=0,
            label=str(digit),
        )
        bottom += values

    ax.set_title(f"{dataset_name}-{distribution}-{strategy_name}")
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Samples")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend(title="Digits", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_name_path = Path(save_name)
    if add_timestamp:
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        output_name = f"{save_name_path.stem}_{timestamp}{save_name_path.suffix or '.png'}"
    else:
        output_name = f"{save_name_path.stem}{save_name_path.suffix or '.png'}"
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Data split seed used: {effective_seed}")
    print(f"Saved figure to: {output_path.resolve()}")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    plot_client_distribution()