import os
import sys
from pathlib import Path

# ── Add src/ to sys.path so `from config import ...` etc. resolve correctly ──
_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import flwr
import torch

from config import (
    CLIENT_NUM_CPUS,
    CLIENT_NUM_GPUS_IF_AVAILABLE,
    DATA_SEED,
    NUM_CLIENTS,
    STRATEGY_NAME,
    StrategyName,
    validate_config,
)
from plot_distribution import plot_distribution
from app.server import recorder, server_app
from strategies.factory import build_strategy


def _setup_paths() -> None:
    """Ensure CWD is project root and both root + src/ are in PYTHONPATH."""
    os.chdir(_ROOT)
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    additions = [str(_ROOT), str(_SRC)]
    new_parts = additions + [p for p in parts if p not in additions]
    os.environ["PYTHONPATH"] = os.pathsep.join(new_parts)


def _get_client_app():
    """Return the ClientApp matching the configured strategy."""
    if STRATEGY_NAME in {StrategyName.FEDXGBBAGGING, StrategyName.FEDXGBCYCLIC}:
        from app.client_xgb import client_app
    else:
        from app.client import client_app
    return client_app


def _build_backend_config() -> dict:
    return {
        "client_resources": {
            "num_cpus": CLIENT_NUM_CPUS,
            "num_gpus": CLIENT_NUM_GPUS_IF_AVAILABLE if torch.cuda.is_available() else 0.0,
        }
    }


def _save_distribution_chart() -> None:
    if recorder.run_dir is not None:
        plot_distribution(
            save_dir=str(recorder.run_dir),
            save_name="distribution.png",
            show=False,
            seed=DATA_SEED,
            add_timestamp=False,
        )


def run() -> None:
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    _setup_paths()
    validate_config()
    build_strategy()  # Surface config issues early, before entering server thread.

    flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=_get_client_app(),
        num_supernodes=NUM_CLIENTS,
        backend_config=_build_backend_config(),
    )

    _save_distribution_chart()


if __name__ == "__main__":
    run()
