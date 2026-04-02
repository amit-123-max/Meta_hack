"""Base grader interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseGrader(ABC):
    """Consumes an episode trajectory and returns a score in [0, 1]."""

    @abstractmethod
    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        """Grade a trajectory.

        Args:
            trajectory: list of step dicts from TrafficSignalEnv.trajectory

        Returns:
            float in [0.0, 1.0]
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        """Clamp-normalise value to [0, 1]."""
        if hi <= lo:
            return 1.0
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    @staticmethod
    def _invert(score: float) -> float:
        """Flip a penalty (higher-is-worse) to a reward (higher-is-better)."""
        return 1.0 - max(0.0, min(1.0, score))

    @staticmethod
    def _extract_stat(trajectory: List[Dict], key: str) -> List[float]:
        """Pull a single scalar metric from each step's state snapshot."""
        vals = []
        for step in trajectory:
            snap = step.get("state_snapshot", {})
            if key in snap:
                vals.append(float(snap[key]))
        return vals
