import os

import flwr
import torch

from config import (
    CLIENT_NUM_CPUS,
    CLIENT_NUM_GPUS_IF_AVAILABLE,
    DATA_SEED,
    NUM_PARTITIONS,
    STRATEGY_NAME,
    StrategyName,
    validate_config,
)
from draw_distribution import plot_client_distribution
from server import recorder, server_app
from strategies.factory import build_strategy


def _is_xgb_strategy() -> bool:
    return STRATEGY_NAME in {StrategyName.FEDXGBBAGGING, StrategyName.FEDXGBCYCLIC}


def _select_client_app():
    if _is_xgb_strategy():
        try:
            import xgboost  # noqa: F401
        except ModuleNotFoundError as ex:
            raise ModuleNotFoundError(
                "XGBoost strategies require package 'xgboost'. "
                "Install it with: pip install xgboost"
            ) from ex

        from client_xgb import client_app as selected_client_app

        return selected_client_app

    from client import client_app as selected_client_app

    return selected_client_app


def run_simulation() -> None:
    # Silence Ray warning about accelerator env var behavior when num_gpus is zero.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    validate_config()
    # Build once in main process to surface compatibility issues early
    # (instead of failing later inside the ServerApp background thread).
    build_strategy()
    selected_client_app = _select_client_app()
    backend_config = {
        "client_resources": {
            "num_cpus": CLIENT_NUM_CPUS,
            "num_gpus": CLIENT_NUM_GPUS_IF_AVAILABLE if torch.cuda.is_available() else 0.0,
        }
    }

    flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=selected_client_app,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )

    if recorder.experiment_dir is not None:
        plot_client_distribution(
            save_dir=str(recorder.experiment_dir),
            save_name="distribution_chart.png",
            show_plot=False,
            seed=DATA_SEED,
            add_timestamp=False,
        )
        print(f"Saved distribution chart to: {recorder.experiment_dir.resolve()}")


if __name__ == "__main__":
    run_simulation()