import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from task import Net, load_centralized_dataset, test_fn


server_app = ServerApp()

fraction_evaluate: float = 0.5
num_rounds: int = 7
lr: float = 0.01

@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # 初始化模型
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # 初始化聚合策略
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # 開始執行策略 跑 num_rounds 次
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # 將最終模型存進硬碟
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


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

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})