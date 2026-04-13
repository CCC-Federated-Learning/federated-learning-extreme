import torch
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from config import LR, NUM_ROUNDS, STRATEGY_NAME, StrategyName, XGB_ETA, XGB_NUM_CLASS, PROXIMAL_MU
from record import Recorder
from strategies.factory import build_strategy
from app.model import Net, load_test_dataloader, eval_model
from app.model_xgb import load_test_data_xgb


server_app = ServerApp()
recorder = Recorder()


def _is_xgb() -> bool:
    return STRATEGY_NAME in {StrategyName.FEDXGBBAGGING, StrategyName.FEDXGBCYCLIC}


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    recorder.start()

    if _is_xgb():
        arrays = ArrayRecord(numpy_ndarrays=[np.array([], dtype=np.uint8)])
        evaluate_fn = _eval_global_xgb
        train_lr = XGB_ETA
    else:
        model = Net()
        arrays = ArrayRecord(torch_state_dict=model.state_dict())
        evaluate_fn = _eval_global
        train_lr = LR

    strategy = build_strategy()

    train_config: dict = {"lr": train_lr}
    if PROXIMAL_MU is not None:
        train_config["proximal-mu"] = PROXIMAL_MU

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(config_dict=train_config),
        num_rounds=NUM_ROUNDS,
        evaluate_fn=evaluate_fn,
    )

    print("\nSaving final model...")
    from pathlib import Path
    Path("results/models").mkdir(parents=True, exist_ok=True)
    if _is_xgb():
        with open("results/models/final_model_xgb.json", "wb") as f:
            f.write(result.arrays["0"].numpy().tobytes())
    else:
        torch.save(result.arrays.to_torch_state_dict(), "results/models/final_model.pt")

    recorder.finalize()


def _eval_global(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global PyTorch model on the centralised test set."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss, acc = eval_model(model, load_test_dataloader(), device)
    recorder.record_round(server_round, acc, loss)
    return MetricRecord(metric_dict={"accuracy": acc, "loss": loss})


def _eval_global_xgb(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global XGBoost booster on the centralised test set."""
    baseline_loss = float(np.log(XGB_NUM_CLASS))
    baseline_acc = float(1.0 / XGB_NUM_CLASS)

    if len(arrays) != 1:
        recorder.record_round(server_round, baseline_acc, baseline_loss)
        return MetricRecord(metric_dict={"accuracy": baseline_acc, "loss": baseline_loss})

    model_bytes = arrays["0"].numpy().tobytes()
    if not model_bytes:
        recorder.record_round(server_round, baseline_acc, baseline_loss)
        return MetricRecord(metric_dict={"accuracy": baseline_acc, "loss": baseline_loss})

    import xgboost as xgb

    x_test, y_test = load_test_data_xgb()
    dtest = xgb.DMatrix(x_test, label=y_test)
    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    probs = booster.predict(dtest)

    eps = 1e-12
    acc = float((np.argmax(probs, axis=1) == y_test).mean())
    loss = float(-np.log(np.clip(probs[np.arange(len(y_test)), y_test], eps, 1.0)).mean())

    recorder.record_round(server_round, acc, loss)
    return MetricRecord(metric_dict={"accuracy": acc, "loss": loss})
