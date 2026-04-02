"""Task 2 — Medium: 2×2 grid with congestion propagation."""
from env.traffic_env import TrafficSignalEnv
from config.task_configs import medium_config


def make_env(seed: int = 42) -> TrafficSignalEnv:
    """Return a configured TrafficSignalEnv for the medium task."""
    cfg = medium_config()
    cfg.sim.seed = seed
    return TrafficSignalEnv(cfg)
