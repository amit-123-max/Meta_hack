"""Base grader interface — shared helpers for all difficulty levels.

Design principles
-----------------
1. Deterministic given the same trajectory.
2. Scores always clamped to [0, 1].
3. Calibration dict is optional; graders fall back to static defaults.
4. Winsorization and trimmed means protect against outliers.
5. Per-step local-action scoring available via _step_process_score().
6. Anti-exploit helpers detect degenerate policies.
"""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseGrader(ABC):
    """Consumes an episode trajectory and returns a score in [0, 1]."""

    def __init__(self, calibration: Optional[Dict[str, Tuple[float, float]]] = None):
        self.calibration = calibration

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        """Grade a trajectory.

        Args:
            trajectory: list of step dicts from TrafficSignalEnv.trajectory

        Returns:
            float in [0.0, 1.0]
        """

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _get_bounds(self, key: str, default_lo: float, default_hi: float) -> Tuple[float, float]:
        """Return calibrated bounds if available, else static defaults."""
        if self.calibration and key in self.calibration:
            lo, hi = self.calibration[key]
            # Sanity: do not return degenerate bounds
            if hi > lo:
                return lo, hi
        return default_lo, default_hi

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
    def _safe_mean(vals: List[float], default: float = 0.0) -> float:
        return statistics.mean(vals) if vals else default

    # ------------------------------------------------------------------
    # Robust statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _winsorize(data: List[float], lo_pct: float = 5.0, hi_pct: float = 95.0) -> List[float]:
        """Clip values to [lo_pct, hi_pct] percentile before aggregation."""
        if len(data) < 4:
            return data
        import numpy as np
        lo = float(np.percentile(data, lo_pct))
        hi = float(np.percentile(data, hi_pct))
        return [max(lo, min(hi, v)) for v in data]

    @staticmethod
    def _trimmed_mean(data: List[float], trim_pct: float = 0.10) -> float:
        """Mean after trimming the bottom and top trim_pct fraction."""
        if not data:
            return 0.0
        n = len(data)
        k = max(0, int(n * trim_pct))
        sdata = sorted(data)
        trimmed = sdata[k: n - k] if k > 0 else sdata
        return float(statistics.mean(trimmed)) if trimmed else float(statistics.mean(sdata))

    def _robust_mean(self, vals: List[float], default: float = 0.0) -> float:
        """Trimmed mean with winsorization — robust against outliers."""
        if not vals:
            return default
        ws = self._winsorize(vals)
        return self._trimmed_mean(ws)

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_stat(trajectory: List[Dict], key: str) -> List[float]:
        """Pull a single scalar metric from each step's state snapshot."""
        vals = []
        for step in trajectory:
            snap = step.get("state_snapshot", {})
            if key in snap:
                vals.append(float(snap[key]))
        return vals

    # ------------------------------------------------------------------
    # Safety gate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_red_rate(trajectory: List[Dict], n_intersections: int) -> float:
        """Fraction of (step × intersection) slots spent fully in ALL_RED."""
        total_slots = max(len(trajectory) * n_intersections, 1)
        all_red_count = 0
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                if inter.get("phase", -1) == 2:  # ALL_RED
                    all_red_count += 1
        return all_red_count / total_slots

    @staticmethod
    def _oscillation_rate(trajectory: List[Dict]) -> float:
        """Fraction of steps where the action changed from the previous step."""
        if len(trajectory) < 2:
            return 0.0
        switches = 0
        prev_phases = None
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            phases = tuple(
                inter.get("phase", -1) for inter in snap.get("intersections", [])
            )
            if prev_phases is not None and phases != prev_phases:
                switches += 1
            prev_phases = phases
        return switches / max(len(trajectory) - 1, 1)

    @staticmethod
    def _starvation_fraction(trajectory: List[Dict], n_intersections: int) -> float:
        """Fraction of intersections where dominant phase > 85% of steps."""
        phase_counts: Dict[int, Dict[int, int]] = {}
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                iid = inter.get("id", 0)
                ph = inter.get("phase", -1)
                if iid not in phase_counts:
                    phase_counts[iid] = {}
                phase_counts[iid][ph] = phase_counts[iid].get(ph, 0) + 1

        if not phase_counts:
            return 0.0
        starved = 0
        for iid, counts in phase_counts.items():
            total = max(sum(counts.values()), 1)
            dominant_frac = max(counts.values()) / total
            if dominant_frac > 0.85:
                starved += 1
        return starved / max(len(phase_counts), 1)

    # ------------------------------------------------------------------
    # Per-step process score
    # ------------------------------------------------------------------

    def _step_process_score(self, step_data: Dict, n_intersections: int) -> float:
        """Score whether the action at this step was locally sensible.

        Returns a value in [0, 1].

        NOTE: We intentionally do NOT check dominant-queue direction here.
        Snapshots capture post-service queue lengths: after serving NS_GREEN the
        NS queues are depleted and EW looks "dominant", which would incorrectly
        penalize a correct action.  Only unambiguous signals are checked:
          1. Emergency lane prioritisation (clear from emergency_type field).
          2. ALL_RED chosen without yellow/emergency justification.
          3. ALL_RED chosen during active spillback (starves the network).
        Default: 0.70 (neutral) when no violation is detected.
        """
        snap = step_data.get("state_snapshot", {})
        inter_list = snap.get("intersections", [])
        if not inter_list:
            return 0.70  # neutral if no info

        checks: List[float] = []

        for inter in inter_list:
            phase          = inter.get("phase", -1)
            emergency      = inter.get("emergency", 0)
            emergency_lane = inter.get("emergency_lane", -1)
            spillback      = inter.get("spillback", 0)

            step_score = 0.70  # neutral baseline

            # 1. Emergency lane check — most important signal
            if emergency > 0 and emergency_lane >= 0:
                expected_phase = 0 if emergency_lane in [0, 1] else 1
                if phase == 2:      # ALL_RED during emergency = bad
                    step_score = 0.0
                elif phase == expected_phase:
                    step_score = 1.0    # correctly serving emergency lane
                else:
                    step_score = 0.15   # serving wrong direction during emergency

            # 2. ALL_RED without emergency or yellow justification
            elif phase == 2 and emergency == 0:
                step_score = 0.05   # very bad — blocks all traffic for no reason

            # 3. ALL_RED during spillback (makes congestion worse)
            elif phase == 2 and spillback > 0:
                step_score = 0.10

            checks.append(step_score)

        return float(statistics.mean(checks)) if checks else 0.70

    def _compute_process_scores(self, trajectory: List[Dict], n_intersections: int) -> List[float]:
        """Compute per-step local process scores for whole trajectory."""
        return [
            self._step_process_score(step_data, n_intersections)
            for step_data in trajectory
        ]

    # ------------------------------------------------------------------
    # Anti-exploit checks
    # ------------------------------------------------------------------

    def _anti_exploit_penalty(
        self,
        trajectory: List[Dict],
        n_intersections: int,
    ) -> float:
        """Return a multiplicative penalty in [0.5, 1.0] for degenerate policies.

        Degenerate patterns penalized:
          - Always ALL_RED (> 60% of steps)
          - Always the same phase (> 90% on one phase, trivially beating starvation gate)
          - Extremely rapid oscillation (> 70% switch rate)
        """
        if not trajectory:
            return 1.0

        all_red = self._all_red_rate(trajectory, n_intersections)
        osc = self._oscillation_rate(trajectory)

        # Collect dominant phase fraction across all intersections
        phase_counts: Dict[int, int] = {}
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                ph = inter.get("phase", -1)
                phase_counts[ph] = phase_counts.get(ph, 0) + 1

        total_phase_slots = max(sum(phase_counts.values()), 1)
        max_single_phase_frac = max(phase_counts.values()) / total_phase_slots if phase_counts else 0.0

        penalty = 1.0
        if all_red > 0.60:
            penalty = min(penalty, 0.3)    # heavy penalty for ALL_RED abuse
        elif all_red > 0.40:
            penalty = min(penalty, 0.6)
        elif all_red > 0.20:
            penalty = min(penalty, 0.85)

        if osc > 0.70:
            penalty = min(penalty, 0.5)   # extreme oscillation
        elif osc > 0.50:
            penalty = min(penalty, 0.75)

        if max_single_phase_frac > 0.92:
            penalty = min(penalty, 0.55)  # lock-in degenerate policy

        return float(max(0.0, min(1.0, penalty)))

    # ------------------------------------------------------------------
    # Jain's fairness (episode-level, robust)
    # ------------------------------------------------------------------

    @staticmethod
    def _jains_fairness_episode(trajectory: List[Dict]) -> float:
        """Episode-level Jain's fairness index across intersections."""
        inter_totals: Dict[int, float] = {}
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                iid = inter.get("id", 0)
                itp = float(inter.get("throughput", 0.0))
                inter_totals[iid] = inter_totals.get(iid, 0.0) + itp

        if len(inter_totals) <= 1:
            return 1.0  # single intersection — trivially fair

        vals = list(inter_totals.values())
        s = sum(vals)
        sq = sum(v * v for v in vals)
        n = len(vals)
        if sq == 0:
            return 0.5  # no throughput anywhere — indeterminate
        return float(min(1.0, (s * s) / (n * sq)))

    # ------------------------------------------------------------------
    # Spillback aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_spillback_rate(trajectory: List[Dict], n_intersections: int) -> float:
        """Mean fraction of intersections spilling per step."""
        rates: List[float] = []
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            inter_list = snap.get("intersections", [])
            if not inter_list:
                continue
            n_i = max(len(inter_list), 1)
            spills = sum(1 for i in inter_list if i.get("spillback", 0) > 0)
            rates.append(spills / n_i)
        return float(statistics.mean(rates)) if rates else 0.0
