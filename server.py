import torch
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from config import LR, NUM_ROUNDS, STRATEGY_NAME, StrategyName, XGB_ETA, XGB_NUM_CLASS
from record import RunRecorder
from strategies.factory import build_strategy
from task import Net, load_centralized_dataset, test_fn
from task_xgb import load_centralized_test_data


server_app = ServerApp()
recorder = RunRecorder()


def _is_xgb_strategy() -> bool:
    return STRATEGY_NAME in {StrategyName.FEDXGBBAGGING, StrategyName.FEDXGBCYCLIC}

@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    recorder.start()

    # 初始化模型
    if _is_xgb_strategy():
        # XGBoost strategies expect a single byte-array (index "0") in ArrayRecord.
        arrays = ArrayRecord([np.array([], dtype=np.uint8)])
        evaluate_fn = global_evaluate_xgb
        train_lr = XGB_ETA
    else:
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())
        evaluate_fn = global_evaluate
        train_lr = LR

    # 初始化聚合策略
    strategy = build_strategy()

    # 開始執行策略 跑 num_rounds 次
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": train_lr}),
        num_rounds=NUM_ROUNDS,
        evaluate_fn=evaluate_fn,
    )

    # 將最終模型存進硬碟
    print("\nSaving final model to disk...")
    if _is_xgb_strategy():
        with open("final_model_xgb.json", "wb") as model_file:
            model_file.write(result.arrays["0"].numpy().tobytes())
    else:
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
    recorder.finalize()


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test_fn(model, test_dataloader, device)
    recorder.record_round(server_round, test_acc, test_loss)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def global_evaluate_xgb(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global XGBoost booster on centralized test data."""
    baseline_loss = float(np.log(XGB_NUM_CLASS))
    baseline_acc = float(1.0 / XGB_NUM_CLASS)

    if len(arrays) != 1:
        recorder.record_round(server_round, baseline_acc, baseline_loss)
        return MetricRecord({"accuracy": baseline_acc, "loss": baseline_loss})

    model_bytes = arrays["0"].numpy().tobytes()
    if not model_bytes:
        recorder.record_round(server_round, baseline_acc, baseline_loss)
        return MetricRecord({"accuracy": baseline_acc, "loss": baseline_loss})

    import xgboost as xgb

    x_test, y_test = load_centralized_test_data()
    dtest = xgb.DMatrix(x_test, label=y_test)

    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    probabilities = booster.predict(dtest)

    eps = 1e-12
    pred_labels = np.argmax(probabilities, axis=1)
    test_acc = float((pred_labels == y_test).mean())
    test_loss = float(
        -np.log(np.clip(probabilities[np.arange(len(y_test)), y_test], eps, 1.0)).mean()
    )

    recorder.record_round(server_round, test_acc, test_loss)
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})