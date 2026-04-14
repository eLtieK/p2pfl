from pathlib import Path
import time
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.utils import set_standalone_settings


def dp_settings() -> None:
    set_standalone_settings()

    Settings.training.VOTE_TIMEOUT = 300
    Settings.training.AGGREGATION_TIMEOUT = 300
    Settings.training.RAY_ACTOR_POOL_SIZE = 4
    

def my_settings() -> None:
    Settings.general.GRPC_TIMEOUT = 400
    Settings.heartbeat.PERIOD = 2
    Settings.heartbeat.TIMEOUT = 400
    Settings.heartbeat.WAIT_CONVERGENCE = 2
    Settings.heartbeat.EXCLUDE_BEAT_LOGS = True
    Settings.gossip.PERIOD = 0
    Settings.gossip.TTL = 10
    Settings.gossip.MESSAGES_PER_PERIOD = 9999999999
    Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED = 10000
    Settings.gossip.MODELS_PERIOD = 2
    Settings.gossip.MODELS_PER_ROUND = 4
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.training.VOTE_TIMEOUT = 600
    Settings.training.AGGREGATION_TIMEOUT = 600
    Settings.training.RAY_ACTOR_POOL_SIZE = 4
    Settings.general.LOG_LEVEL = "INFO"
    logger.set_level(Settings.general.LOG_LEVEL)  # Refresh (maybe already initialize
    

def flatten_config(cfg: dict):
    """
    Flatten config dạng:
    {"dp": {"epsilon": 10}} -> {"dp_epsilon": 10}
    """
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat

def build_output_dir(base_cfg, extra_cfgs=None, prefix="exp", root_dir="results"):
    """
    base_cfg: config chung (nodes, rounds,...)
    extra_cfgs: danh sách config bổ sung (DP / Dual DP)
    prefix: tên experiment (dp / dual_dp / baseline)
    root_dir: thư mục gốc (results / results_attacks / ...)
    """

    parts = [
        prefix,
        f"n{base_cfg['nodes']}",
        f"r{base_cfg['rounds']}",
        f"e{base_cfg['epochs']}",
        f"bs{base_cfg['batch_size']}",
    ]

    if extra_cfgs:
        for extra_cfg in extra_cfgs:
            flat = flatten_config(extra_cfg)

            for k, v in flat.items():
                short_k = (
                    k.replace("dual_mode_", "")
                     .replace("dp_", "")
                )
                parts.append(f"{short_k}{v}")

    name = "_".join(parts)

    return Path(root_dir) / name