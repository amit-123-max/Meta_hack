"""Hard-task grader — emergency + partial obs + weather noise.

Architecture: hybrid gate × (process + outcome)
================================================

  final_score = gate_factor
                × (W_PROCESS * mean(process_scores)
                   + W_OUTCOME * outcome_score)

  gate_factor = safety_gate(emergency_neglect, starvation, spillback)
              × anti_exploit_penalty

  outcome_score = weighted_sum(
      0.28 * throughput_score,
      0.28 * emergency_quality_score,
      0.22 * spillback_score,
      0.14 * fairness_score,
      0.08 * smoothness_score,
  )

Emergency quality score
-----------------------
  Uses EXPLICIT event log from trajectory["state_snapshot"]["emergency_events"]
  if available (new simulator format). Falls back to state-machine reconstruction
  from snapshot diffs for backward compatibility.

  Per-event score = response_curve(latency, type)
    Ambulance ideal ≤ 4 steps, bad ≥ 15 steps
    Fire      ideal ≤ 6 steps, bad ≥ 20 steps
    Police    ideal ≤ 8 steps, bad ≥ 28 steps

  Unserved emergencies score 0 and trigger the neglect gate.

Emergency neglect gate
----------------------
  Any emergency that expires without ever receiving green service
  triggers a 0.5× gate factor per neglect, floored at 0.15.

Process / outcome mix
---------------------
  W_PROCESS = 0.35
  W_OUTCOME = 0.65

Difficulty ceiling
------------------
  DIFFICULTY_CEIL = 0.58   (hard < medium at equivalent policies)
  Applied via soft cap (same tanh-like function as MediumGrader).

Target ranges
-------------
  Rule-based baseline: ≈ 0.22–0.38
  Good LLM policy:     ≈ 0.35–0.50
  Random policy:       ≈ 0.10–0.25
  Score always in [0, 1].
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from graders.base_grader import BaseGrader

# Normalization defaults
_TP_DEFAULT_LO    = 0.0
_TP_DEFAULT_HI    = 14.0   # 4 intersections, higher arrival rate
_QUEUE_DEFAULT_LO = 0.0
_QUEUE_DEFAULT_HI = 8.0
_SPILLBACK_DEFAULT_HI = 0.80
_SWITCH_DEFAULT_HI    = 0.30

N_INTERSECTIONS = 4

# Emergency response thresholds: (ideal_steps, bad_steps) per emergency type
_EMERG_THRESHOLDS: Dict[int, Tuple[int, int]] = {
    3: (4,  15),   # Ambulance: ideal ≤ 4, bad ≥ 15
    2: (6,  20),   # Fire:      ideal ≤ 6, bad ≥ 20
    1: (8,  28),   # Police:    ideal ≤ 8, bad ≥ 28
}

# Process / outcome blend
W_PROCESS = 0.35
W_OUTCOME = 0.65

# Outcome sub-weights (sum = 1.0)
W_TP            = 0.28
W_EMERG_QUALITY = 0.28
W_SPILLBACK     = 0.22
W_FAIRNESS      = 0.14
W_SMOOTH        = 0.08

# Soft difficulty ceiling
DIFFICULTY_CEIL = 0.58

# When no emergencies occur: neutral gate (no penalty)
_NO_EMERGENCY_GATE = 1.0


def _soft_cap(score: float, ceiling: float) -> float:
    """Soft cap: approach ceiling asymptotically."""
    if ceiling <= 0.0:
        return 0.0
    return ceiling * (1.0 - math.exp(-3.0 * score / ceiling))


def _resp_score(latency_steps: int, em_type: int) -> float:
    """Map response latency to [0, 1]. Shorter is better."""
    ideal, bad = _EMERG_THRESHOLDS.get(em_type, (8, 28))
    if latency_steps <= ideal:
        return 1.0
    if latency_steps >= bad:
        return 0.0
    t = (latency_steps - ideal) / max(bad - ideal, 1)
    return float(math.exp(-3.0 * t))


class HardGrader(BaseGrader):
    """Deterministic grader for Task 3 (Hard).

    Target baseline (rule-based): ≈ 0.22–0.38.
    Emergency quality uses explicit simulator event log when available.
    """

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)

        # ------------------------------------------------------------------
        # 1. Parse emergency events (prefer explicit log; fallback to diffs)
        # ------------------------------------------------------------------
        em_scores: List[float] = []
        neglect_count: int = 0

        # Try explicit event log from the last snapshot (accumulates all events)
        last_snap = trajectory[-1].get("state_snapshot", {})
        explicit_events: List[Dict] = last_snap.get("emergency_events", [])

        if explicit_events:
            for ev in explicit_events:
                latency = int(ev.get("latency_steps", 999))
                etype   = int(ev.get("etype", 0))
                served  = bool(ev.get("served", False))
                if served:
                    em_scores.append(_resp_score(latency, etype))
                else:
                    em_scores.append(0.0)
                    neglect_count += 1
        else:
            # Backward-compatible state-machine reconstruction from snapshot diffs
            em_state: Dict[str, Dict] = {}
            for step_data in trajectory:
                snap   = step_data.get("state_snapshot", {})
                s_step = int(snap.get("step", 0))

                for inter in snap.get("intersections", []):
                    iid      = inter.get("id", 0)
                    em_type  = int(inter.get("emergency", 0))
                    em_lane  = int(inter.get("emergency_lane", -1))
                    phase    = int(inter.get("phase", 2))
                    key      = f"{iid}_em"

                    green_lanes = {0: [0, 1], 1: [2, 3]}.get(phase, [])

                    if em_type > 0:
                        if key not in em_state:
                            em_state[key] = {
                                "arrival": s_step,
                                "em_type": em_type,
                                "done": False,
                            }
                        info = em_state[key]
                        if not info["done"] and em_lane in green_lanes:
                            waited = s_step - info["arrival"]
                            em_scores.append(_resp_score(waited, em_type))
                            info["done"] = True
                    elif key in em_state:
                        info = em_state.pop(key)
                        if not info["done"]:
                            em_scores.append(0.0)
                            neglect_count += 1

            # Flush remaining open emergencies
            for info in em_state.values():
                if not info["done"]:
                    em_scores.append(0.0)
                    neglect_count += 1

        # Emergency gate: neglect decays gate toward 0.15 floor
        if em_scores:
            avg_em = float(statistics.mean(em_scores))
            # Neglect gate: 0.5× per neglect, floored at 0.15
            neglect_gate = max(0.15, 1.0 - 0.40 * neglect_count)
            emergency_gate = avg_em * neglect_gate
        else:
            emergency_gate = _NO_EMERGENCY_GATE  # no emergencies → neutral
            neglect_gate   = 1.0

        # Emergency quality score (for outcome weighting)
        emergency_quality_score = float(statistics.mean(em_scores)) if em_scores else 1.0

        # ------------------------------------------------------------------
        # 2. Safety gate (all-red, starvation, spillback, oscillation)
        # ------------------------------------------------------------------
        all_red = self._all_red_rate(trajectory, N_INTERSECTIONS)
        osc     = self._oscillation_rate(trajectory)
        starv   = self._starvation_fraction(trajectory, N_INTERSECTIONS)
        spill   = self._mean_spillback_rate(trajectory, N_INTERSECTIONS)

        gate = emergency_gate  # start from emergency gate

        if all_red > 0.50:
            gate *= 0.20
        elif all_red > 0.30:
            gate *= 0.55
        elif all_red > 0.15:
            gate *= 0.80

        if starv > 0.0:
            n_starved = int(round(starv * N_INTERSECTIONS))
            gate *= max(0.35, 1.0 - 0.18 * n_starved)

        if spill > 0.75:
            gate *= 0.65
        elif spill > 0.55:
            gate *= 0.82

        if osc > 0.70:
            gate *= 0.55
        elif osc > 0.50:
            gate *= 0.80

        exploit_penalty = self._anti_exploit_penalty(trajectory, N_INTERSECTIONS)
        gate = max(0.0, min(1.0, gate * exploit_penalty))

        if gate == 0.0:
            return 0.0

        # ------------------------------------------------------------------
        # 3. Process score
        # ------------------------------------------------------------------
        process_scores = self._compute_process_scores(trajectory, N_INTERSECTIONS)
        process_score  = self._robust_mean(process_scores, default=0.5)

        # ------------------------------------------------------------------
        # 4. Outcome score
        # ------------------------------------------------------------------
        throughputs:  List[float] = []
        worst_queues: List[float] = []

        for step_data in trajectory:
            snap = step_data.get("state_snapshot", {})
            throughputs.append(float(snap.get("global_throughput", 0.0)))
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
        q_s_raw = self._invert(
            self._normalise(self._robust_mean(worst_queues, q_hi), q_lo, q_hi)
        )

        sp_lo, sp_hi = self._get_bounds("spillback", 0.0, _SPILLBACK_DEFAULT_HI)
        sp_s = self._invert(self._normalise(spill, sp_lo, sp_hi))

        fair_s = self._jains_fairness_episode(trajectory)

        final_sw = int(last_snap.get("phase_switches", 0))
        switch_rate = final_sw / max(n_steps * N_INTERSECTIONS, 1)
        smooth_s = self._invert(
            self._normalise(switch_rate, 0.0, _SWITCH_DEFAULT_HI)
        )

        outcome_score = max(0.0, min(1.0,
            W_TP            * tp_s
            + W_EMERG_QUALITY * emergency_quality_score
            + W_SPILLBACK     * sp_s
            + W_FAIRNESS      * fair_s
            + W_SMOOTH        * smooth_s
        ))

        # ------------------------------------------------------------------
        # 5. Combined score with soft difficulty ceiling
        # ------------------------------------------------------------------
        raw    = W_PROCESS * process_score + W_OUTCOME * outcome_score
        capped = _soft_cap(raw, DIFFICULTY_CEIL)
        final  = gate * capped
        return float(max(0.0, min(1.0, final)))
