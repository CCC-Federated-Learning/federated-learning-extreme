import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.differential_privacy import compute_adaptive_clip_model_update
from flwr.common.differential_privacy_constants import KEY_CLIPPING_NORM, KEY_NORM_BIT

from config import (
    BATCH_SIZE,
    CLIENT_NUM_CPUS,
    DATA_DISTRIBUTION,
    DATA_SEED,
    DIRICHLET_ALPHA,
    LOCAL_EPOCHS,
)
from app.task import Net, load_data, train_fn, test_fn

client_app = ClientApp()


def _configure_torch_threads() -> None:
    """Align PyTorch thread usage with per-client CPU allocation."""
    num_threads = max(1, int(CLIENT_NUM_CPUS))
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(num_threads)

@client_app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    _configure_torch_threads()

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Keep a CPU snapshot of server parameters for DP clipping mods.
    server_state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        BATCH_SIZE,
        distribution=DATA_DISTRIBUTION,
        dirichlet_alpha=DIRICHLET_ALPHA,
        seed=DATA_SEED,
    )

    config = msg.content["config"]
    proximal_mu = float(config.get("proximal-mu", 0.0))
    global_params = (
        [param.detach().clone() for param in model.parameters()]
        if proximal_mu > 0
        else None
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        LOCAL_EPOCHS,
        config["lr"],
        device,
        proximal_mu=proximal_mu,
        global_params=global_params,
    )

    # Construct and return reply Message
    if KEY_CLIPPING_NORM in config:
        clipping_norm = float(config[KEY_CLIPPING_NORM])
        client_state_dict = model.state_dict()

        server_params = [tensor.numpy() for tensor in server_state_dict.values()]
        client_params = [tensor.detach().cpu().numpy() for tensor in client_state_dict.values()]

        # This applies clipping in-place to client_params and returns norm_bit.
        norm_bit = compute_adaptive_clip_model_update(
            client_params, server_params, clipping_norm
        )

        clipped_state_dict = {}
        for (key, ref_tensor), clipped_arr in zip(client_state_dict.items(), client_params):
            clipped_state_dict[key] = torch.from_numpy(clipped_arr).to(
                device=ref_tensor.device, dtype=ref_tensor.dtype
            )
        model.load_state_dict(clipped_state_dict)

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    if KEY_CLIPPING_NORM in config:
        metrics[KEY_NORM_BIT] = float(norm_bit)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    _configure_torch_threads()

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(
        partition_id,
        num_partitions,
        BATCH_SIZE,
        distribution=DATA_DISTRIBUTION,
        dirichlet_alpha=DIRICHLET_ALPHA,
        seed=DATA_SEED,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)