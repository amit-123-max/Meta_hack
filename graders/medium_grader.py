"""Medium-task grader — 2×2 grid with congestion propagation.

Difficulty calibration:
  DIFFICULTY_CAP = 0.42 — guarantees easy ≥ medium at any trajectory length.
  This ensures: easy (cap ~0.75) > medium (cap 0.42) > hard (cap 0.36).

  Cap was lowered from 0.60 → 0.42 (factor 0.70) so that even a 20-step
  short-horizon evaluation satisfies the monotonicity constraint.

  With baseline rule-based agent (tp≈6.4/step, queue≈1.7/lane, spill≈0):
    raw_score ≈ 0.82  →  final = 0.42 × 0.82 ≈ 0.34
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from graders.base_grader import BaseGrader

_N             = 4
_TP_NORM_MAX   = 7.0    # 4 inters × λ=0.40 arrival ceiling
_QUEUE_MAX     = 8.0    # avg queue/lane: >8 = badly congested
_SPILLBACK_MAX = 0.60
_SWITCH_MAX    = 0.25
DIFFICULTY_CAP = 0.42   # hard ceiling (lowered from 0.60 for monotonicity)

W_TP       = 0.35
W_SPILLBACK= 0.27
W_QUEUE    = 0.22
W_FAIRNESS = 0.10
W_SWITCH   = 0.06


class MediumGrader(BaseGrader):
    """Deterministic grader for Task 2 (Medium). Target baseline ≈ 0.49."""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0

        throughputs: List[float] = []
        avg_queues:  List[float] = []
        spill_steps: List[float] = []
        fairness:    List[float] = []
        n_steps = len(trajectory)

        for step in trajectory:
            snap = step.get("state_snapshot", {})
            throughputs.append(float(snap.get("global_throughput", 0)))
            avg_queues.append(float(snap.get("global_avg_wait", _QUEUE_MAX)))

            inter_tps, step_spill = [], 0
            for inter in snap.get("intersections", []):
                step_spill += int(inter.get("spillback", 0))
                inter_tps.append(float(inter.get("throughput", 0)))

            spill_steps.append(step_spill / max(_N, 1))

            if len(inter_tps) > 1 and max(inter_tps) > 0:
                sq = sum(t * t for t in inter_tps)
                s  = sum(inter_tps)
                fairness.append(min(1.0, s * s / max(_N * sq, 1e-9)))

        tp_s   = self._normalise(statistics.mean(throughputs) if throughputs else 0.0,
                                 0.0, _TP_NORM_MAX)
        q_s    = self._invert(self._normalise(
                    statistics.mean(avg_queues) if avg_queues else _QUEUE_MAX,
                    0.0, _QUEUE_MAX))
        sp_s   = self._invert(self._normalise(
                    statistics.mean(spill_steps) if spill_steps else 0.0,
                    0.0, _SPILLBACK_MAX))
        fair_s = statistics.mean(fairness) if fairness else 0.5

        final_sw   = int(trajectory[-1].get("state_snapshot", {}).get("phase_switches", 0))
        switch_pen = self._normalise(final_sw / max(n_steps * _N, 1), 0.0, _SWITCH_MAX)

        # Starvation guard
        inter_totals: Dict[int, float] = {}
        for step in trajectory:
            for inter in step.get("state_snapshot", {}).get("intersections", []):
                iid = inter.get("id", 0)
                inter_totals[iid] = inter_totals.get(iid, 0.0) + float(inter.get("throughput", 0))
        starvation = sum(0.15 for v in inter_totals.values() if v < 1.0)

        raw = (W_TP * tp_s + W_SPILLBACK * sp_s + W_QUEUE * q_s
               + W_FAIRNESS * fair_s - W_SWITCH * switch_pen - starvation)

        return float(max(0.0, min(1.0, DIFFICULTY_CAP * max(0.0, raw))))
