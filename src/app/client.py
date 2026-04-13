import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.differential_privacy import compute_adaptive_clip_model_update
from flwr.common.differential_privacy_constants import KEY_CLIPPING_NORM, KEY_NORM_BIT

from config import BATCH_SIZE, CLIENT_NUM_CPUS, DATA_DISTRIBUTION, DATA_SEED, DIRICHLET_ALPHA, LOCAL_EPOCHS
from app.model import Net, load_client_data, train_model, eval_model

client_app = ClientApp()
_threads_set = False


def _configure_threads() -> None:
    global _threads_set
    if _threads_set:
        return
    n = max(1, int(CLIENT_NUM_CPUS))
    torch.set_num_threads(n)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass
    _threads_set = True


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    _configure_threads()

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Snapshot server params before training (needed for DP delta and FedProx).
    server_params_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    client_id = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]
    trainloader, _ = load_client_data(
        client_id, num_clients, BATCH_SIZE,
        distribution=DATA_DISTRIBUTION, dirichlet_alpha=DIRICHLET_ALPHA, seed=DATA_SEED,
    )

    config = msg.content["config"]
    proximal_mu = float(config.get("proximal-mu", 0.0))
    global_params = (
        [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    )

    train_loss = train_model(
        model, trainloader, LOCAL_EPOCHS, config["lr"], device,
        proximal_mu=proximal_mu, global_params=global_params,
    )

    # Apply differential privacy clipping if the server sent a clipping norm.
    if KEY_CLIPPING_NORM in config:
        clipping_norm = float(config[KEY_CLIPPING_NORM])
        state = model.state_dict()
        s_params = [t.numpy() for t in server_params_cpu.values()]
        c_params = [t.detach().cpu().numpy() for t in state.values()]
        norm_bit = compute_adaptive_clip_model_update(c_params, s_params, clipping_norm)
        clipped = {
            k: torch.from_numpy(arr).to(device=ref.device, dtype=ref.dtype)
            for (k, ref), arr in zip(state.items(), c_params)
        }
        model.load_state_dict(clipped)

    metrics: dict = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    if KEY_CLIPPING_NORM in config:
        metrics[KEY_NORM_BIT] = float(norm_bit)

    content = RecordDict(records={
        "arrays": ArrayRecord(torch_state_dict=model.state_dict()),
        "metrics": MetricRecord(metric_dict=metrics),
    })
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    _configure_threads()

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    client_id = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]
    _, valloader = load_client_data(
        client_id, num_clients, BATCH_SIZE,
        distribution=DATA_DISTRIBUTION, dirichlet_alpha=DIRICHLET_ALPHA, seed=DATA_SEED,
    )

    loss, acc = eval_model(model, valloader, device)

    content = RecordDict(records={
        "metrics": MetricRecord(metric_dict={
            "eval_loss": loss,
            "eval_acc": acc,
            "num-examples": len(valloader.dataset),
        }),
    })
    return Message(content=content, reply_to=msg)
