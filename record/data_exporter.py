import csv
import json
from pathlib import Path

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
    STRATEGY_NAME,
)


class DataExporter:
    """Export run metrics and metadata files."""

    @staticmethod
    def save_json(experiment_dir: Path, records: list[dict[str, float | int]]) -> Path:
        output_path = experiment_dir / "run_metrics.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)
        return output_path

    @staticmethod
    def save_csv(experiment_dir: Path, records: list[dict[str, float | int]]) -> Path:
        output_path = experiment_dir / "run_metrics.csv"
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["round", "accuracy", "loss"])
            writer.writeheader()
            writer.writerows(records)
        return output_path

    @staticmethod
    def save_metadata(
        experiment_dir: Path,
        run_id: str,
        model_name: str,
        elapsed_seconds: float,
    ) -> Path:
        metadata_path = experiment_dir / "metadata.txt"
        content = (
            f"run_id = {run_id}\n"
            f"title = {DATASET_NAME}-{DATA_DISTRIBUTION}-{STRATEGY_NAME}\n"
            f"model_name = {model_name}\n"
            f"dataset_name = {DATASET_NAME}\n"
            f"data_distribution = {DATA_DISTRIBUTION}\n"
            f"strategy = {STRATEGY_NAME}\n"
            f"num_rounds = {NUM_ROUNDS}\n"
            f"local_epochs = {LOCAL_EPOCHS}\n"
            f"batch_size = {BATCH_SIZE}\n"
            f"learning_rate = {LR}\n"
            f"fraction_evaluate = {FRACTION_EVALUATE}\n"
            f"dirichlet_alpha = {DIRICHLET_ALPHA}\n"
            f"data_seed = {DATA_SEED}\n"
            f"num_partitions = {NUM_PARTITIONS}\n"
            f"elapsed_seconds = {elapsed_seconds:.2f}\n"
        )
        metadata_path.write_text(content, encoding="utf-8")
        return metadata_path
