"""Rule-based baseline agent for TrafficSignalEnv.

Strategy:
1. Emergency override — if any lane has an emergency vehicle, determine
   which phase serves that lane and switch to it immediately.
   Priority: ambulance > fire > police.
2. Pressure-based phase selection — choose the phase whose green lanes
   have the highest total queue pressure (queue_length × weight).
3. Phase extension — if the current phase has higher queue pressure than
   any alternative and the phase timer hasn't exceeded max, hold it.
4. Anti-oscillation — enforce minimum phase duration before switching.

The agent reads from TrafficObservation.metadata (no CNN required).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from env.schemas import (
    EmergencyType,
    PhaseEnum,
    TrafficAction,
    TrafficObservation,
)
from env.simulator import PHASE_GREEN_LANES

# Emergency priority weight amplifier
_EMERGENCY_WEIGHT = {
    EmergencyType.NONE:      0.0,
    EmergencyType.POLICE:    5.0,
    EmergencyType.FIRE:      10.0,
    EmergencyType.AMBULANCE: 20.0,
}

# Phase → lane indices it serves (indices 0=N,1=S,2=E,3=W)
_PHASE_LANES = {
    0: [0, 1],  # NS_GREEN
    1: [2, 3],  # EW_GREEN
    2: [],       # ALL_RED
}


class RuleBasedAgent:
    """Deterministic, reproducible rule-based policy.

    Works for any number of intersections by computing independent
    phase decisions per intersection.
    """

    def __init__(
        self,
        n_intersections: int,
        min_phase_steps: int = 5,
        max_phase_steps: int = 40,
    ) -> None:
        self.n_intersections  = n_intersections
        self.min_phase_steps  = min_phase_steps
        self.max_phase_steps  = max_phase_steps
        self._phase_timers    = [0] * n_intersections
        self._current_phases  = [0] * n_intersections  # all start NS_GREEN

    def reset(self) -> None:
        self._phase_timers   = [0] * self.n_intersections
        self._current_phases = [0] * self.n_intersections

    def act(self, obs: TrafficObservation) -> TrafficAction:
        """Choose actions from metadata in TrafficObservation."""
        meta = obs.metadata  # (n_inter, 11)
        n    = min(self.n_intersections, meta.shape[0])
        phases: List[int] = []

        for i in range(n):
            phase = self._decide_intersection(i, meta[i])
            phases.append(phase)

        # Pad if needed
        while len(phases) < self.n_intersections:
            phases.append(-1)

        return TrafficAction(phase_indices=phases)

    # ------------------------------------------------------------------
    # Per-intersection decision
    # ------------------------------------------------------------------

    def _decide_intersection(self, iid: int, meta_row: np.ndarray) -> int:
        """Decide phase for a single intersection from its metadata row.

        Metadata layout (index → feature):
          0-3: queue fractions (q0..q3)
          4:   phase_norm
          5:   phase_timer_norm
          6:   yellow_remaining_norm
          7:   emergency_type_norm
          8:   emergency_lane_norm
          9:   weather_norm
          10:  spillback_flag
        """
        q    = meta_row[:4].tolist()          # [0..1] fractions
        # Decode current phase & timer (rough decode from normalised)
        n_phases = max(len(PhaseEnum) - 1, 1)
        cur_phase = int(round(meta_row[4] * n_phases))
        cur_phase = max(0, min(2, cur_phase))  # clamp to valid

        timer_norm  = float(meta_row[5])
        yellow_norm = float(meta_row[6])
        emerg_norm  = float(meta_row[7])
        emerg_lane_idx = int(round(float(meta_row[8]) * 4)) - 1  # -1 if none

        # Still in yellow transition — hold current
        if yellow_norm > 0.05:
            self._phase_timers[iid] += 1
            return -1  # hold / noop

        # Reconstruct timer from norm
        approx_timer = int(timer_norm * 40)  # 40 = phase_duration_max approx
        self._phase_timers[iid] = approx_timer

        # 1. Emergency override logic
        if emerg_norm > 0.01 and emerg_lane_idx >= 0:
            em_type_val = int(round(emerg_norm * (len(EmergencyType) - 1)))
            em_type     = EmergencyType(max(0, min(3, em_type_val)))

            # Find which phase serves the emergency lane
            for p_idx, lane_group in _PHASE_LANES.items():
                if emerg_lane_idx in lane_group:
                    if approx_timer >= self.min_phase_steps or cur_phase != p_idx:
                        self._current_phases[iid] = p_idx
                        return p_idx

        # 2. Pressure-based selection (if past min phase duration)
        if approx_timer >= self.min_phase_steps or approx_timer >= self.max_phase_steps:
            ns_pressure = q[0] + q[1]   # North + South
            ew_pressure = q[2] + q[3]   # East  + West

            # Check for spillback amplification
            spill = float(meta_row[10]) if len(meta_row) > 10 else 0.0
            if spill > 0.5:
                # amplify whichever direction is more overloaded
                ns_pressure *= 1.5 if ns_pressure > ew_pressure else 1.0
                ew_pressure *= 1.5 if ew_pressure >= ns_pressure else 1.0

            best_phase = 0 if ns_pressure >= ew_pressure else 1
            if best_phase != cur_phase:
                self._current_phases[iid] = best_phase
                return best_phase

        # 3. Hold current phase
        return -1  # noop
