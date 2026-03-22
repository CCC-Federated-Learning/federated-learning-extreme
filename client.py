import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from config import BATCH_SIZE, DATA_DISTRIBUTION, DIRICHLET_ALPHA, LOCAL_EPOCHS, DATA_SEED
from task import Net, load_data, train_fn, test_fn

client_app = ClientApp()

@client_app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
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
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

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