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
from app.model_xgb import load_client_data_xgb

client_app = ClientApp()


def _decode_booster(arrays: ArrayRecord) -> xgb.Booster | None:
    if len(arrays) != 1:
        raise ValueError("XGBoost ArrayRecord must contain exactly one entry")
    model_bytes = arrays["0"].numpy().tobytes()
    if not model_bytes:
        return None
    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    return booster


def _encode_booster(booster: xgb.Booster) -> ArrayRecord:
    raw = booster.save_raw(raw_format="json")
    return ArrayRecord(numpy_ndarrays=[np.frombuffer(raw, dtype=np.uint8)])


def _build_params(lr: float) -> dict:
    return {
        "objective": XGB_OBJECTIVE,
        "num_class": XGB_NUM_CLASS,
        "eta": lr,
        "max_depth": XGB_MAX_DEPTH,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "lambda": XGB_REG_LAMBDA,
        "tree_method": "hist",
        "nthread": max(1, int(CLIENT_NUM_CPUS)),
        "eval_metric": "mlogloss",
    }


def _compute_metrics(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(labels) == 0:
        return 0.0, 0.0
    eps = 1e-12
    acc = float((np.argmax(probs, axis=1) == labels).mean())
    loss = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())
    return loss, acc


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    client_id = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]

    x_train, y_train, _, _ = load_client_data_xgb(
        client_id=client_id, num_clients=num_clients,
        distribution=DATA_DISTRIBUTION, dirichlet_alpha=DIRICHLET_ALPHA, seed=DATA_SEED,
    )

    lr = float(msg.content["config"].get("lr", XGB_ETA))
    global_booster = _decode_booster(msg.content["arrays"])
    # Cyclic: continue adding trees to the global model.
    # Bagging: train independent trees from scratch each round.
    xgb_model = global_booster if STRATEGY_NAME == StrategyName.FEDXGBCYCLIC else None

    evals_result: dict = {}
    booster = xgb.train(
        params=_build_params(lr),
        dtrain=xgb.DMatrix(x_train, label=y_train),
        num_boost_round=XGB_NUM_LOCAL_ROUND,
        evals=[(xgb.DMatrix(x_train, label=y_train), "train")],
        evals_result=evals_result,
        verbose_eval=False,
        xgb_model=xgb_model,
    )

    train_loss = float(evals_result.get("train", {}).get("mlogloss", [0.0])[-1])

    content = RecordDict(records={
        "arrays": _encode_booster(booster),
        "metrics": MetricRecord(metric_dict={
            "train_loss": train_loss,
            "num-examples": len(y_train),
        }),
    })
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    client_id = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]

    _, _, x_test, y_test = load_client_data_xgb(
        client_id=client_id, num_clients=num_clients,
        distribution=DATA_DISTRIBUTION, dirichlet_alpha=DIRICHLET_ALPHA, seed=DATA_SEED,
    )

    booster = _decode_booster(msg.content["arrays"])
    if booster is None or len(y_test) == 0:
        content = RecordDict(records={
            "metrics": MetricRecord(metric_dict={
                "eval_loss": 0.0, "eval_acc": 0.0, "num-examples": len(y_test),
            }),
        })
        return Message(content=content, reply_to=msg)

    probs = booster.predict(xgb.DMatrix(x_test, label=y_test))
    loss, acc = _compute_metrics(probs, y_test)

    content = RecordDict(records={
        "metrics": MetricRecord(metric_dict={
            "eval_loss": loss, "eval_acc": acc, "num-examples": len(y_test),
        }),
    })
    return Message(content=content, reply_to=msg)
