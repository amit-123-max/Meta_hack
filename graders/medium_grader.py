"""Medium-task grader — 4-intersection grid, congestion propagation.

Architecture: hybrid gate × (process + outcome)
================================================

  final_score = gate_factor
                × (W_PROCESS * mean(process_scores)
                   + W_OUTCOME * outcome_score)

  gate_factor = safety_gate(all_red, starvation, spillback_overflow)
              × anti_exploit_penalty

  outcome_score = weighted_sum(
      0.38 * throughput_score,
      0.26 * spillback_score,
      0.20 * queue_score,
      0.10 * fairness_score,   # Jain's, episode-level
      0.06 * smoothness_score,
  )

Safety gates (multiplicative)
------------------------------
  ALL_RED rate > 50% → gate = 0.20
  ALL_RED rate > 30% → gate ×= 0.55
  Any intersection starved (< 2% expected throughput) → gate ×= 0.70 per starved
  Mean spillback rate > 0.75 → gate ×= 0.60

Process / outcome mix
---------------------
  W_PROCESS = 0.25
  W_OUTCOME = 0.75

Difficulty ceiling
------------------
  DIFFICULTY_CEIL = 0.72   (soft upper bound on final score)
  Applied via tanh-like stretch: score is mapped through a soft cap
  rather than a flat multiplier, preserving gradient in [0, CEIL].

Target ranges
-------------
  Rule-based baseline: ≈ 0.35–0.50
  Good LLM policy:     ≈ 0.45–0.60
  Random policy:       ≈ 0.15–0.32
  Score always in [0, 1].
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from graders.base_grader import BaseGrader

# Normalization defaults
_TP_DEFAULT_LO    = 0.0
_TP_DEFAULT_HI    = 10.0   # 4 intersections, good policy
_QUEUE_DEFAULT_LO = 0.0
_QUEUE_DEFAULT_HI = 8.0
_SPILLBACK_DEFAULT_HI = 0.75
_SWITCH_DEFAULT_HI    = 0.30

N_INTERSECTIONS = 4

# Process / outcome blend
W_PROCESS = 0.25
W_OUTCOME = 0.75

# Outcome sub-weights (sum = 1.0)
W_TP       = 0.38
W_SPILLBACK = 0.26
W_QUEUE    = 0.20
W_FAIRNESS = 0.10
W_SMOOTH   = 0.06

# Soft difficulty ceiling
DIFFICULTY_CEIL = 0.72


def _soft_cap(score: float, ceiling: float) -> float:
    """Soft cap: approach ceiling asymptotically, full signal below it."""
    if ceiling <= 0.0:
        return 0.0
    # scores near ceiling approach it smoothly; scores << ceiling are linear
    # f(x) = ceil * (1 - exp(-3x/ceil))  — maps [0,∞) onto [0, ceil)
    return ceiling * (1.0 - math.exp(-3.0 * score / ceiling))


class MediumGrader(BaseGrader):
    """Deterministic grader for Task 2 (Medium).

    Target baseline (rule-based): ≈ 0.35–0.50.
    Exploit-resistant via safety gate + anti-exploit layer.
    """

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)

        # ------------------------------------------------------------------
        # 1. Safety gate
        # ------------------------------------------------------------------
        all_red = self._all_red_rate(trajectory, N_INTERSECTIONS)
        osc     = self._oscillation_rate(trajectory)
        starv   = self._starvation_fraction(trajectory, N_INTERSECTIONS)
        spill   = self._mean_spillback_rate(trajectory, N_INTERSECTIONS)

        gate = 1.0
        if all_red > 0.50:
            gate *= 0.20
        elif all_red > 0.30:
            gate *= 0.55
        elif all_red > 0.15:
            gate *= 0.80

        # Starvation gate (per starved intersection)
        if starv > 0.0:
            n_starved = int(round(starv * N_INTERSECTIONS))
            gate *= max(0.40, 1.0 - 0.15 * n_starved)

        # Severe spillback gate
        if spill > 0.75:
            gate *= 0.60
        elif spill > 0.55:
            gate *= 0.80

        if osc > 0.70:
            gate *= 0.55
        elif osc > 0.50:
            gate *= 0.80

        exploit_penalty = self._anti_exploit_penalty(trajectory, N_INTERSECTIONS)
        gate = max(0.0, min(1.0, gate * exploit_penalty))

        if gate == 0.0:
            return 0.0

        # ------------------------------------------------------------------
        # 2. Process score
        # ------------------------------------------------------------------
        process_scores = self._compute_process_scores(trajectory, N_INTERSECTIONS)
        process_score  = self._robust_mean(process_scores, default=0.5)

        # ------------------------------------------------------------------
        # 3. Outcome score
        # ------------------------------------------------------------------
        throughputs:  List[float] = []
        worst_queues: List[float] = []

        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            throughputs.append(float(snap.get("global_throughput", 0.0)))

            q_hi_default = _QUEUE_DEFAULT_HI
            step_worst_q = 0.0
            for inter in snap.get("intersections", []):
                qs = inter.get("queues", [])
                if qs:
                    local_worst = max(float(q) for q in qs)
                    step_worst_q = max(step_worst_q, local_worst)
            worst_queues.append(step_worst_q)

        tp_lo, tp_hi = self._get_bounds("tp", _TP_DEFAULT_LO, _TP_DEFAULT_HI)
        tp_s = self._normalise(self._robust_mean(throughputs, 0.0), tp_lo, tp_hi)

        q_lo, q_hi = self._get_bounds("queue", _QUEUE_DEFAULT_LO, _QUEUE_DEFAULT_HI)
        q_s = self._invert(
            self._normalise(self._robust_mean(worst_queues, q_hi), q_lo, q_hi)
        )

        sp_lo, sp_hi = self._get_bounds("spillback", 0.0, _SPILLBACK_DEFAULT_HI)
        sp_s = self._invert(
            self._normalise(spill, sp_lo, sp_hi)
        )

        fair_s = self._jains_fairness_episode(trajectory)

        final_sw = int(trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0))
        switch_rate = final_sw / max(n_steps * N_INTERSECTIONS, 1)
        smooth_s = self._invert(
            self._normalise(switch_rate, 0.0, _SWITCH_DEFAULT_HI)
        )

        outcome_score = max(0.0, min(1.0,
            W_TP        * tp_s
            + W_SPILLBACK * sp_s
            + W_QUEUE    * q_s
            + W_FAIRNESS * fair_s
            + W_SMOOTH   * smooth_s
        ))

        # ------------------------------------------------------------------
        # 4. Combined score with soft difficulty ceiling
        # ------------------------------------------------------------------
        raw = W_PROCESS * process_score + W_OUTCOME * outcome_score
        capped = _soft_cap(raw, DIFFICULTY_CEIL)
        final  = gate * capped
        return float(max(0.0, min(1.0, final)))
