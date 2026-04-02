"""Task 3 — Hard: emergency vehicles + partial obs + weather noise."""
from env.traffic_env import TrafficSignalEnv
from config.task_configs import hard_config


def make_env(seed: int = 42) -> TrafficSignalEnv:
    """Return a configured TrafficSignalEnv for the hard task."""
    cfg = hard_config()
    cfg.sim.seed = seed
    return TrafficSignalEnv(cfg)
