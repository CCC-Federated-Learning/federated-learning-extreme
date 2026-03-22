import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from config import LR, NUM_ROUNDS
from record import RunRecorder
from strategies.factory import build_strategy
from task import Net, load_centralized_dataset, test_fn


server_app = ServerApp()
recorder = RunRecorder()

@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    recorder.start()

    # 初始化模型
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # 初始化聚合策略
    strategy = build_strategy()

    # 開始執行策略 跑 num_rounds 次
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": LR}),
        num_rounds=NUM_ROUNDS,
        evaluate_fn=global_evaluate,
    )

    # 將最終模型存進硬碟
    print("\nSaving final model to disk...")
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