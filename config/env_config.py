"""Configuration dataclasses for TrafficSignalEnv."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SimConfig:
    """Low-level simulation parameters."""
    # Time
    dt: float = 1.0                   # seconds per step
    max_steps: int = 500

    # Phases per intersection: (phase_name, green_lanes_mask)
    # A phase is a subset of lanes that are green simultaneously.
    # Defined per-intersection; overridden by TaskConfig if needed.
    phase_duration_min: int = 5       # min steps a phase must hold
    phase_duration_max: int = 60      # max steps before forced switch
    yellow_duration: int = 3          # steps of yellow between phases

    # Traffic arrival (Poisson λ vehicles/step per lane)
    arrival_rate_base: float = 0.3
    arrival_rate_noise: float = 0.1   # uniform noise on top of base

    # Vehicle parameters
    max_queue_per_lane: int = 40      # overflow → spillback penalty
    discharge_rate: int = 3           # vehicles released per green step

    # Emergency vehicle
    emergency_prob_per_step: float = 0.0
    emergency_clear_steps: int = 10   # steps to pass through intersection

    # Congestion propagation (medium/hard)
    spillback_threshold: float = 0.85  # fraction of max_queue → spillback
    propagation_fraction: float = 0.30 # fraction of overflow pushed upstream

    # Partial observability
    occlusion_prob: float = 0.0       # probability camera is blocked this step

    # Weather
    weather_change_prob: float = 0.0  # prob of weather change per step
    seed: int = 42


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    n_intersections: int = 1
    lanes_per_intersection: int = 4
    grid_shape: Tuple[int, int] = (1, 1)   # (rows, cols)
    adjacency: Optional[List[List[int]]] = None  # None → auto from grid

    # Observation image
    image_size: Tuple[int, int] = (84, 84)
    frame_stack: int = 4               # number of frames stacked
    include_metadata: bool = True

    # Task tag (drives grader selection)
    task_id: str = "easy"

    sim: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self) -> None:
        if self.adjacency is None:
            self.adjacency = self._build_grid_adjacency()

    def _build_grid_adjacency(self) -> List[List[int]]:
        """Build adjacency list from grid_shape."""
        rows, cols = self.grid_shape
        assert rows * cols == self.n_intersections, (
            f"grid_shape {self.grid_shape} inconsistent with "
            f"n_intersections={self.n_intersections}"
        )
        adj: List[List[int]] = [[] for _ in range(self.n_intersections)]
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if r > 0:
                    adj[idx].append((r - 1) * cols + c)
                if r < rows - 1:
                    adj[idx].append((r + 1) * cols + c)
                if c > 0:
                    adj[idx].append(r * cols + (c - 1))
                if c < cols - 1:
                    adj[idx].append(r * cols + (c + 1))
        return adj
