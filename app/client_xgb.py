import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from config import (
    CLIENT_NUM_CPUS,
    DATA_DISTRIBUTION,
    DATA_SEED,
    DIRICHLET_ALPHA,
    STRATEGY_NAME,
    StrategyName,
    XGB_COLSAMPLE_BYTREE,
    XGB_ETA,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_NUM_CLASS,
    XGB_NUM_LOCAL_ROUND,
    XGB_OBJECTIVE,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
)
from app.task_xgb import load_partition_data

client_app = ClientApp()


def _load_booster_from_arrays(arrays: ArrayRecord) -> xgb.Booster | None:
    """Decode global booster bytes from ArrayRecord."""
    if len(arrays) != 1:
        raise ValueError("XGBoost strategies require exactly one array in ArrayRecord")

    model_bytes = arrays["0"].numpy().tobytes()
    if not model_bytes:
        return None

    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    return booster


def _booster_to_arrayrecord(booster: xgb.Booster) -> ArrayRecord:
    """Encode an XGBoost booster into Flower ArrayRecord."""
    model_bytes = booster.save_raw(raw_format="json")
    model_array = np.frombuffer(model_bytes, dtype=np.uint8)
    return ArrayRecord([model_array])


def _build_xgb_params(train_lr: float) -> dict:
    num_threads = max(1, int(CLIENT_NUM_CPUS))
    return {
        "objective": XGB_OBJECTIVE,
        "num_class": XGB_NUM_CLASS,
        "eta": train_lr,
        "max_depth": XGB_MAX_DEPTH,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "lambda": XGB_REG_LAMBDA,
        "tree_method": "hist",
        "nthread": num_threads,
        "eval_metric": "mlogloss",
    }


def _multiclass_metrics(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute (logloss, accuracy) for multiclass probabilities."""
    if len(labels) == 0:
        return 0.0, 0.0

    eps = 1e-12
    pred_labels = np.argmax(probabilities, axis=1)
    accuracy = float((pred_labels == labels).mean())

    clipped = np.clip(probabilities[np.arange(len(labels)), labels], eps, 1.0)
    loss = float(-np.log(clipped).mean())
    return loss, accuracy


@client_app.train()
def train(msg: Message, context: Context):
    """Train an XGBoost model on local tabular data and return booster bytes."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    x_train, y_train, _, _ = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        distribution=DATA_DISTRIBUTION,
        dirichlet_alpha=DIRICHLET_ALPHA,
        seed=DATA_SEED,
    )

    dtrain = xgb.DMatrix(x_train, label=y_train)
    train_config = msg.content["config"]
    train_lr = float(train_config.get("lr", XGB_ETA))
    params = _build_xgb_params(train_lr)

    incoming_arrays = msg.content["arrays"]
    incoming_booster = _load_booster_from_arrays(incoming_arrays)

    # Bagging trains independent trees; cyclic continues from current global booster.
    xgb_model = None
    if STRATEGY_NAME == StrategyName.FEDXGBCYCLIC:
        xgb_model = incoming_booster

    evals_result: dict = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=XGB_NUM_LOCAL_ROUND,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
        verbose_eval=False,
        xgb_model=xgb_model,
    )

    train_loss = 0.0
    if "train" in evals_result and "mlogloss" in evals_result["train"]:
        train_loss = float(evals_result["train"]["mlogloss"][-1])

    model_record = _booster_to_arrayrecord(booster)
    metric_record = MetricRecord(
        {
            "train_loss": train_loss,
            "num-examples": len(y_train),
        }
    )
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate received XGBoost booster on local test partition."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    _, _, x_test, y_test = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        distribution=DATA_DISTRIBUTION,
        dirichlet_alpha=DIRICHLET_ALPHA,
        seed=DATA_SEED,
    )

    booster = _load_booster_from_arrays(msg.content["arrays"])
    if booster is None or len(y_test) == 0:
        metric_record = MetricRecord(
            {"eval_loss": 0.0, "eval_acc": 0.0, "num-examples": len(y_test)}
        )
        return Message(content=RecordDict({"metrics": metric_record}), reply_to=msg)

    dtest = xgb.DMatrix(x_test, label=y_test)
    probabilities = booster.predict(dtest)

    eval_loss, eval_acc = _multiclass_metrics(probabilities, y_test)
    metric_record = MetricRecord(
        {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(y_test),
        }
    )
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
