"""Easy-task grader — single intersection, normal traffic.

Architecture: hybrid gate × (process + outcome)
================================================

  final_score = gate_factor
                × (W_PROCESS * mean(process_scores)
                   + W_OUTCOME * outcome_score)

  gate_factor = safety_gate(all_red_abuse, starvation, oscillation)
              × anti_exploit_penalty

  outcome_score = weighted_sum(
      0.45 * throughput_score,   # primary — must discharge vehicles
      0.30 * queue_score,        # worst-lane tail-risk
      0.15 * improvement_score,  # late-episode progress
      0.10 * smoothness_score,   # anti-oscillation
  )

Safety gates (multiplicative)
------------------------------
  ALL_RED rate > 60% → gate = 0.20
  ALL_RED rate > 40% → gate = 0.55
  Starvation fraction > 0%  → gate ×= 0.55 (one dominant phase > 85%)
  Oscillation rate > 70%    → gate ×= 0.50

Process / outcome mix
---------------------
  W_PROCESS = 0.30   (local per-step action quality)
  W_OUTCOME = 0.70   (trajectory-level outcome metrics)

Calibration
-----------
  Uses calibrated bounds from calibration dict if available.
  Static defaults:
    tp    : [0.0, 4.0]   (single intersection, discharge=3, 2 green lanes)
    queue : [0.0, 8.0]

Target ranges
-------------
  Rule-based baseline: ≈ 0.45–0.65
  Good LLM policy:     ≈ 0.60–0.78
  Random policy:       ≈ 0.20–0.40
  Score always in [0, 1].
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from graders.base_grader import BaseGrader

# Normalization defaults (used when calibration is absent)
_TP_DEFAULT_LO    = 0.0
_TP_DEFAULT_HI    = 1.5    # realistic upper bound: arrival_rate=0.30 × 4 lanes ≈ 1.2/step
_QUEUE_DEFAULT_LO = 0.0
_QUEUE_DEFAULT_HI = 8.0
_SWITCH_DEFAULT_HI = 0.25  # switch rate reference

# Process / outcome blend weights
W_PROCESS = 0.30
W_OUTCOME = 0.70

# Outcome sub-weights (must sum to 1.0)
W_TP      = 0.45
W_QUEUE   = 0.30
W_IMPROVE = 0.15
W_SMOOTH  = 0.10


class EasyGrader(BaseGrader):
    """Deterministic grader for Task 1 (Easy).

    Target baseline (rule-based): ≈ 0.45–0.65.
    Exploit-resistant via safety gate + anti-exploit layer.
    """

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)
        n_inters = 1  # Easy = single intersection

        # ------------------------------------------------------------------
        # 1. Safety gate
        # ------------------------------------------------------------------
        all_red = self._all_red_rate(trajectory, n_inters)
        osc     = self._oscillation_rate(trajectory)

        gate = 1.0
        if all_red > 0.60:
            gate *= 0.20
        elif all_red > 0.40:
            gate *= 0.55
        elif all_red > 0.20:
            gate *= 0.80

        # Starvation gate for easy (single intersection, short episodes).
        # Threshold raised to 0.92 — with Poisson arrivals over ~20 steps,
        # one direction naturally dominates >85% of steps for a good policy.
        # We only penalize true lock-in (>92%) where both directions are never
        # balanced AND queues on the ignored side are building up.
        phase_counts: Dict[int, int] = {}
        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                ph = inter.get("phase", -1)
                phase_counts[ph] = phase_counts.get(ph, 0) + 1
        if phase_counts:
            total_ph = max(sum(phase_counts.values()), 1)
            dominant_frac = max(phase_counts.values()) / total_ph
            if dominant_frac > 0.92:   # truly degenerate lock-in
                gate *= 0.72           # soft penalty (was 0.55 — too harsh)

        if osc > 0.70:
            gate *= 0.50
        elif osc > 0.50:
            gate *= 0.75

        # Anti-exploit multiplicative factor
        exploit_penalty = self._anti_exploit_penalty(trajectory, n_inters)
        gate = max(0.0, min(1.0, gate * exploit_penalty))

        if gate == 0.0:
            return 0.0

        # ------------------------------------------------------------------
        # 2. Process score (per-step local action quality)
        # ------------------------------------------------------------------
        process_scores = self._compute_process_scores(trajectory, n_inters)
        process_score  = self._robust_mean(process_scores, default=0.5)

        # ------------------------------------------------------------------
        # 3. Outcome score
        # ------------------------------------------------------------------
        throughputs:  List[float] = []
        worst_queues: List[float] = []

        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            throughputs.append(float(snap.get("global_throughput", 0.0)))

            lane_queues: List[float] = []
            for inter in snap.get("intersections", []):
                for q in inter.get("queues", []):
                    lane_queues.append(float(q))
            q_hi = self._get_bounds("queue", _QUEUE_DEFAULT_LO, _QUEUE_DEFAULT_HI)[1]
            worst_q = max(lane_queues) if lane_queues else q_hi
            worst_queues.append(worst_q)

        # Calibrated throughput normalization
        tp_lo, tp_hi = self._get_bounds("tp", _TP_DEFAULT_LO, _TP_DEFAULT_HI)
        tp_score = self._normalise(self._robust_mean(throughputs, 0.0), tp_lo, tp_hi)

        # Queue score (lower is better)
        q_lo, q_hi = self._get_bounds("queue", _QUEUE_DEFAULT_LO, _QUEUE_DEFAULT_HI)
        queue_score = self._invert(
            self._normalise(self._robust_mean(worst_queues, q_hi), q_lo, q_hi)
        )

        # Improvement score: late-episode vs early-episode throughput
        half = max(n_steps // 2, 1)
        early_tp = self._safe_mean(throughputs[:half])
        late_tp  = self._safe_mean(throughputs[half:]) if n_steps > half else early_tp
        # Centered at 0.5; clamped [0,1]
        improve_score = max(0.0, min(1.0,
            (late_tp - early_tp) / max(tp_hi - tp_lo, 1e-6) + 0.5
        ))

        # Smoothness: penalise high switch rate
        final_sw = int(
            trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0)
        )
        switch_rate = final_sw / max(n_steps, 1)
        smooth_score = self._invert(
            self._normalise(switch_rate, 0.0, _SWITCH_DEFAULT_HI)
        )

        outcome_score = (
            W_TP      * tp_score
            + W_QUEUE   * queue_score
            + W_IMPROVE * improve_score
            + W_SMOOTH  * smooth_score
        )
        outcome_score = max(0.0, min(1.0, outcome_score))

        # ------------------------------------------------------------------
        # 4. Combined score
        # ------------------------------------------------------------------
        raw = W_PROCESS * process_score + W_OUTCOME * outcome_score
        final = gate * raw
        return float(max(0.0, min(1.0, final)))
