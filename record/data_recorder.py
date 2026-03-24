import time
from datetime import datetime
from pathlib import Path

from config import RES_DIR, TIMESTAMP_FORMAT


class DataRecorder:
    """Collect per-round metrics and manage run lifecycle."""

    def __init__(self, res_dir: str = RES_DIR, model_name: str = "Net"):
        self.res_dir = Path(res_dir)
        self.experiment_dir: Path | None = None
        self.run_id = ""
        self.records: list[dict[str, float | int]] = []
        self.start_perf: float | None = None
        self.model_name = model_name

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

    def get_elapsed_seconds(self) -> float:
        if self.start_perf is None:
            return 0.0
        return time.perf_counter() - self.start_perf
