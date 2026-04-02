"""Easy-task grader — single intersection, normal traffic.

Calibration grounded in actual arrival physics:
  - 4 lanes × λ=0.30 = 1.2 vehicles/step total arriving
  - Throughput ≤ arrival → realistic max TP ≈ 1.4 vehicles/step
  - Avg queue per lane ≈ 1.5–4.0 for a good agent (low arrival rate)
  - Expected baseline score: ~0.62–0.70 (highest of the three tasks,
    because this is the simplest — single intersection, no emergencies)

Score formula:
  score = 0.45 * throughput_score
        + 0.40 * queue_score
        - 0.15 * switch_penalty
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from graders.base_grader import BaseGrader

# --------------------------------------------------------------------------
# Per-step calibration — grounded in arrival physics not theoretical max
# --------------------------------------------------------------------------
# Arrival: 4 lanes × λ=0.30 = 1.2 vehicles/step across the intersection.
# A perfect agent discharges all arrivals → TP ≈ 1.2/step.
# We set _GOOD_TP = 1.0 (achievable) and _BAD_TP = 0 (nothing moves).
_TP_GOOD     = 1.0    # realistic well-run throughput
_TP_NORM_MAX = 1.4    # absolute cap for normalisation (slightly above good)

# Average queue per lane: with λ=0.30, phases ~10s each, queue stays low.
# Good agent: ~1.5–2.5 vehicles/lane. Bad agent: ≥6 vehicles/lane.
_QUEUE_GOOD  = 2.0    # target average queue per lane
_QUEUE_MAX   = 7.0    # represents a badly congested lane

# Switch rate: number of phase changes / total steps.
# Oscillating every step → 1.0; stable control → near 0.
_SWITCH_RATE_MAX = 0.20   # 1 switch per 5 steps = clearly oscillatory

W_TP     = 0.45
W_QUEUE  = 0.40
W_SWITCH = 0.15


class EasyGrader(BaseGrader):
    """Deterministic grader for Task 1 (Easy).

    Expected score range with rule-based baseline: ~0.62–0.70.
    This should be the HIGHEST score of the three tasks.
    """

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        throughputs: List[float] = []
        avg_queues:  List[float] = []
        phase_counts: Dict[int, int] = {}

        for step in trajectory:
            snap = step.get("state_snapshot", {})
            throughputs.append(float(snap.get("global_throughput", 0)))
            avg_queues.append(float(snap.get("global_avg_wait", _QUEUE_MAX)))

            for inter in snap.get("intersections", []):
                p = inter.get("phase", -1)
                phase_counts[p] = phase_counts.get(p, 0) + 1

        n_steps = len(trajectory)

        # --- Throughput: normalise against physics-based ceiling ---
        avg_tp   = statistics.mean(throughputs) if throughputs else 0.0
        tp_score = self._normalise(avg_tp, 0.0, _TP_NORM_MAX)

        # --- Queue: lower is better; 0 queue → perfect, _QUEUE_MAX → 0 ---
        mean_queue  = statistics.mean(avg_queues) if avg_queues else _QUEUE_MAX
        queue_score = self._invert(self._normalise(mean_queue, 0.0, _QUEUE_MAX))

        # --- Switch penalty: final total switches / steps = rate ---
        final_switches = int(
            trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0)
        )
        switch_rate    = final_switches / max(n_steps, 1)
        switch_penalty = self._normalise(switch_rate, 0.0, _SWITCH_RATE_MAX)

        # --- Degenerate guard: one phase dominates >92% → starvation ---
        if phase_counts:
            dominant_frac = max(phase_counts.values()) / max(sum(phase_counts.values()), 1)
            if dominant_frac > 0.92:
                tp_score    *= 0.6
                queue_score *= 0.6

        score = (
            W_TP    * tp_score
            + W_QUEUE * queue_score
            - W_SWITCH * switch_penalty
        )
        return float(max(0.0, min(1.0, score)))
