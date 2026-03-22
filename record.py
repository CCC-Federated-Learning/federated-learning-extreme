import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

# Use a non-GUI backend to avoid Tkinter/thread issues in Flower worker threads.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE,
    DATASET_NAME,
    DATA_DISTRIBUTION,
    DATA_SEED,
    DIRICHLET_ALPHA,
    FRACTION_EVALUATE,
    LOCAL_EPOCHS,
    LR,
    NUM_PARTITIONS,
    NUM_ROUNDS,
    RES_DIR,
    STRATEGY_NAME,
    TIMESTAMP_FORMAT,
)


class RunRecorder:
    """Record per-round metrics and export result files/charts."""

    def __init__(self, res_dir: str = RES_DIR):
        self.res_dir = Path(res_dir)
        self.experiment_dir = None
        self.run_id = ""
        self.records = []
        self.start_perf = None
        self.model_name = "Net"

    def start(self) -> None:
        self.run_id = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.records = []
        self.start_perf = time.perf_counter()
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.res_dir / self.run_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def record_round(self, round_idx: int, accuracy: float, loss: float) -> None:
        if not self.run_id:
            self.start()
        self.records.append(
            {
                "round": int(round_idx),
                "accuracy": float(accuracy),
                "loss": float(loss),
            }
        )

    def _save_json(self) -> Path:
        output_path = self.experiment_dir / "run_metrics.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        return output_path

    def _save_csv(self) -> Path:
        output_path = self.experiment_dir / "run_metrics.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
            writer.writeheader()
            writer.writerows(self.records)
        return output_path

    def _save_acc_chart(self) -> Path:
        chart_path = self.experiment_dir / "acc_round.png"

        rounds = [item["round"] for item in self.records]
        accs = [item["accuracy"] for item in self.records]

        # If accuracy is stored in [0, 1], convert it to percentage for readability.
        if accs and max(accs) <= 1.0:
            accs = [value * 100.0 for value in accs]

        fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=120)
        fig.patch.set_facecolor("#E6E6E6")
        ax.set_facecolor("#E6E6E6")

        ax.plot(
            rounds,
            accs,
            color="#0077B6",
            marker="o",
            markersize=5,
            markeredgecolor="#003049",
            linewidth=2.2,
            label="Model Accuracy",
        )

        run_title = f"{DATASET_NAME}-{DATA_DISTRIBUTION}-{STRATEGY_NAME}"
        ax.set_title(run_title, fontsize=18, pad=10)
        ax.set_xlabel("Training iteration", fontsize=16)
        ax.set_ylabel("Accuracy %", fontsize=16)

        ax.set_xticks(rounds)
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

    def _get_elapsed_seconds(self) -> float:
        elapsed_seconds = 0.0
        if self.start_perf is not None:
            elapsed_seconds = time.perf_counter() - self.start_perf
        return elapsed_seconds

    def _save_metadata_file(self) -> Path:
        elapsed_seconds = self._get_elapsed_seconds()
        metadata_path = self.experiment_dir / "metadata.txt"

        content = (
            f"run_id={self.run_id}\n"
            f"title={DATASET_NAME}-{DATA_DISTRIBUTION}-{STRATEGY_NAME}\n"
            f"model_name={self.model_name}\n"
            f"dataset_name={DATASET_NAME}\n"
            f"data_distribution={DATA_DISTRIBUTION}\n"
            f"strategy={STRATEGY_NAME}\n"
            f"num_rounds={NUM_ROUNDS}\n"
            f"local_epochs={LOCAL_EPOCHS}\n"
            f"batch_size={BATCH_SIZE}\n"
            f"learning_rate={LR}\n"
            f"fraction_evaluate={FRACTION_EVALUATE}\n"
            f"dirichlet_alpha={DIRICHLET_ALPHA}\n"
            f"data_seed={DATA_SEED}\n"
            f"num_partitions={NUM_PARTITIONS}\n"
            f"elapsed_seconds={elapsed_seconds:.2f}\n"
        )
        metadata_path.write_text(content, encoding="utf-8")
        return metadata_path

    def finalize(self):
        if not self.records:
            print("No round records to save.")
            return None

        json_path = self._save_json()
        csv_path = self._save_csv()
        chart_path = self._save_acc_chart()
        metadata_path = self._save_metadata_file()

        print(f"Saved run records to: {json_path.resolve()}")
        print(f"Saved run records to: {csv_path.resolve()}")
        print(f"Saved acc chart to: {chart_path.resolve()}")
        print(f"Saved run info to: {metadata_path.resolve()}")

        return json_path, csv_path, chart_path, metadata_path
