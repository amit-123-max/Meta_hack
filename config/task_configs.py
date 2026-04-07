"""Task-specific configuration presets."""
from __future__ import annotations

from config.env_config import EnvConfig, SimConfig


def easy_config() -> EnvConfig:
    """Task 1: single intersection, normal traffic."""
    return EnvConfig(
        n_intersections=1,
        lanes_per_intersection=4,
        grid_shape=(1, 1),
        image_size=(84, 84),
        frame_stack=4,
        include_metadata=True,
        task_id="easy",
        sim=SimConfig(
            max_steps=20,            # capped for LLM efficiency
            arrival_rate_base=0.30,
            arrival_rate_noise=0.10,
            emergency_prob_per_step=0.0,
            occlusion_prob=0.0,
            weather_change_prob=0.0,
            seed=42,
        ),
    )


def medium_config() -> EnvConfig:
    """Task 2: 2×2 grid with congestion propagation."""
    return EnvConfig(
        n_intersections=4,
        lanes_per_intersection=4,
        grid_shape=(2, 2),
        image_size=(84, 84),
        frame_stack=4,
        include_metadata=True,
        task_id="medium",
        sim=SimConfig(
            max_steps=20,            # capped for LLM efficiency
            arrival_rate_base=0.40,
            arrival_rate_noise=0.15,
            emergency_prob_per_step=0.0,
            occlusion_prob=0.0,
            weather_change_prob=0.0,
            spillback_threshold=0.80,
            propagation_fraction=0.35,
            seed=42,
        ),
    )


def hard_config() -> EnvConfig:
    """Task 3: emergency vehicles + partial obs + weather noise."""
    return EnvConfig(
        n_intersections=4,
        lanes_per_intersection=4,
        grid_shape=(2, 2),
        image_size=(84, 84),
        frame_stack=4,
        include_metadata=True,
        task_id="hard",
        sim=SimConfig(
            max_steps=20,            # capped for LLM efficiency
            arrival_rate_base=0.45,
            arrival_rate_noise=0.20,
            emergency_prob_per_step=0.015,    # ~1 emergency per ~67 steps
            occlusion_prob=0.08,
            weather_change_prob=0.02,
            spillback_threshold=0.75,
            propagation_fraction=0.40,
            seed=42,
        ),
    )
