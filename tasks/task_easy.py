"""Task 1 — Easy: single intersection, normal traffic."""
from env.traffic_env import TrafficSignalEnv
from config.task_configs import easy_config


def make_env(seed: int = 42) -> TrafficSignalEnv:
    """Return a configured TrafficSignalEnv for the easy task."""
    cfg = easy_config()
    cfg.sim.seed = seed
    return TrafficSignalEnv(cfg)
