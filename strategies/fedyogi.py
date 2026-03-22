from flwr.serverapp.strategy import FedYogi

from config import BETA_1, BETA_2, FRACTION_EVALUATE, SERVER_ETA, SERVER_ETA_L, SERVER_TAU


def build_fedyogi():
    """Build Flower FedYogi strategy from project config."""
    return FedYogi(
        fraction_evaluate=FRACTION_EVALUATE,
        eta=SERVER_ETA,
        eta_l=SERVER_ETA_L,
        beta_1=BETA_1,
        beta_2=BETA_2,
        tau=SERVER_TAU,
    )
