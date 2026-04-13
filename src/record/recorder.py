import time
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR, STRATEGY_NAME, TIMESTAMP_FORMAT
from .exporter import Exporter
from .chart import plot_accuracy_chart


class Recorder:
    """Collect per-round metrics and export results when the run ends."""

    def __init__(self, results_dir: str = RESULTS_DIR, model_name: str = "Net"):
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.run_id: str = ""
        self.run_dir: Path | None = None
        self.rounds: list[dict[str, float | int]] = []
        self._t0: float | None = None

    def start(self) -> None:
        self.run_id = datetime.now().strftime(TIMESTAMP_FORMAT) + f"-{STRATEGY_NAME}"
        self.rounds = []
        self._t0 = time.perf_counter()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def record_round(self, round_idx: int, accuracy: float, loss: float) -> None:
        if not self.run_id:
            self.start()
        self.rounds.append(
            {"round": int(round_idx), "accuracy": float(accuracy), "loss": float(loss)}
        )

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    def finalize(self) -> None:
        if not self.rounds:
            print("No round records to save.")
            return
        if self.run_dir is None:
            raise RuntimeError("Recorder.start() was never called")

        json_path  = Exporter.save_json(self.run_dir, self.rounds)
        csv_path   = Exporter.save_csv(self.run_dir, self.rounds)
        chart_path = plot_accuracy_chart(self.run_dir, self.rounds)
        meta_path  = Exporter.save_metadata(
            self.run_dir,
            run_id=self.run_id,
            model_name=self.model_name,
            elapsed_seconds=self.elapsed(),
        )

        print(f"Saved metrics JSON  : {json_path.resolve()}")
        print(f"Saved metrics CSV   : {csv_path.resolve()}")
        print(f"Saved accuracy chart: {chart_path.resolve()}")
        print(f"Saved metadata      : {meta_path.resolve()}")

    # Alias kept for compatibility with run.py which reads experiment_dir.
    @property
    def experiment_dir(self) -> Path | None:
        return self.run_dir
