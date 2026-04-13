from config import RES_DIR
from .data_exporter import DataExporter
from .data_recorder import DataRecorder
from .draw_acc_chart import draw_acc_chart


class RunRecorder:
    """Record per-round metrics and export result files/charts."""

    def __init__(self, res_dir: str = RES_DIR):
        self._recorder = DataRecorder(res_dir=res_dir, model_name="Net")

    def start(self) -> None:
        self._recorder.start()

    def record_round(self, round_idx: int, accuracy: float, loss: float) -> None:
        self._recorder.record_round(round_idx, accuracy, loss)

    def finalize(self):
        if not self._recorder.records:
            print("No round records to save.")
            return None

        if self._recorder.experiment_dir is None:
            raise RuntimeError("Recorder was not started correctly")

        json_path = DataExporter.save_json(
            self._recorder.experiment_dir,
            self._recorder.records,
        )
        csv_path = DataExporter.save_csv(
            self._recorder.experiment_dir,
            self._recorder.records,
        )
        chart_path = draw_acc_chart(
            self._recorder.experiment_dir,
            self._recorder.records,
        )
        metadata_path = DataExporter.save_metadata(
            self._recorder.experiment_dir,
            run_id=self._recorder.run_id,
            model_name=self._recorder.model_name,
            elapsed_seconds=self._recorder.get_elapsed_seconds(),
        )

        print(f"Saved run records to: {json_path.resolve()}")
        print(f"Saved run records to: {csv_path.resolve()}")
        print(f"Saved acc chart to: {chart_path.resolve()}")
        print(f"Saved run info to: {metadata_path.resolve()}")

        return json_path, csv_path, chart_path, metadata_path

    @property
    def experiment_dir(self):
        return self._recorder.experiment_dir
