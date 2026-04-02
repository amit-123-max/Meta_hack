"""Dense multi-objective reward shaping for TrafficSignalEnv."""
from __future__ import annotations

from typing import List

from env.schemas import (
    EmergencyType,
    IntersectionState,
    PhaseEnum,
    TrafficReward,
    TrafficState,
)
from env.simulator import PHASE_GREEN_LANES

# ---------------------------------------------------------------------------
# Reward weights — tuned for dense, informative, abuse-resistant shaping
# ---------------------------------------------------------------------------
W_THROUGHPUT    =  0.40
W_QUEUE         = -0.25
W_WAIT          = -0.20
W_SWITCH        = -0.08
W_EMERGENCY     =  0.60
W_SPILLBACK     = -0.30
W_STARVATION    = -0.20
W_FAIRNESS      =  0.15

# Emergency priority multipliers (ambulance >> fire > police)
EMERGENCY_MULTIPLIER = {
    EmergencyType.POLICE:    1.0,
    EmergencyType.FIRE:      2.0,
    EmergencyType.AMBULANCE: 4.0,
}
MAX_EMERGENCY_REWARD = 2.0    # cap per step
MAX_QUEUE            = 40.0   # normaliser
MAX_WAIT_PER_STEP    = 40.0   # vehicles × dt
D_SWITCH_THRESHOLD   = 3      # phase_timer below this → oscillation penalty


def compute_reward(
    state: TrafficState,
    prev_state: TrafficState | None,
    step_stats: dict,
    max_queue: int,
) -> TrafficReward:
    """Compute dense reward from current and previous states.

    Args:
        state:      Current TrafficState.
        prev_state: Previous TrafficState (None on first step).
        step_stats: Raw stat dict returned by simulator.step().
        max_queue:  SimConfig.max_queue_per_lane.

    Returns:
        TrafficReward with per-component values; .total gives scalar.
    """
    reward = TrafficReward()

    total_tp         = 0.0
    total_queue      = 0.0
    total_wait       = 0.0
    total_spillback  = 0.0
    total_starvation = 0.0
    emergency_reward = 0.0
    fairness_reward  = 0.0
    switch_penalty   = 0.0
    n_intersections  = len(state.intersections)
    n_lanes          = n_intersections * 4

    for istat in state.intersections:
        # --- Throughput bonus ---
        tp_norm = istat.total_throughput / max(1, n_lanes * 3)  # 3 = discharge_rate
        total_tp += tp_norm

        # --- Queue penalty ---
        for lane in istat.lanes:
            q_norm = lane.queue_length / max(max_queue, 1)
            total_queue += q_norm

        # --- Wait penalty ---
        wait_norm = istat.total_wait / max(MAX_WAIT_PER_STEP * n_intersections, 1)
        total_wait += wait_norm

        # --- Spillback penalty ---
        total_spillback += istat.spillback_count

        # --- Starvation penalty ---
        for lane in istat.lanes:
            # Starvation timer approx via wait: lanes that have been waiting
            # very long (> 30 steps) with non-zero queue
            if lane.wait_time > 30 * 40:  # arbitrary long wait threshold
                total_starvation += lane.queue_length / max(max_queue, 1)

        # --- Emergency reward / penalty ---
        if istat.emergency_active != EmergencyType.NONE:
            green_lanes = PHASE_GREEN_LANES.get(istat.current_phase, [])
            em_lane     = istat.emergency_lane
            mult = EMERGENCY_MULTIPLIER.get(istat.emergency_active, 1.0)
            if em_lane in green_lanes:
                emergency_reward += mult * 1.0    # emergency gets green → big bonus
            else:
                emergency_reward -= mult * 0.5    # emergency waiting → penalty

        # --- Phase-switch oscillation penalty ---
        # If phase was switched AND timer was very low → oscillation
        for istat_raw in step_stats.get("intersections", []):
            if istat_raw.get("intersection_id") == istat.intersection_id:
                if istat_raw.get("phase_switched") and istat.phase_timer < D_SWITCH_THRESHOLD:
                    switch_penalty += 1.0
                break

    # --- Fairness bonus (Jain's fairness index on throughput per intersection) ---
    throughputs = [i.total_throughput for i in state.intersections]
    if n_intersections > 1 and max(throughputs) > 0:
        tp_arr_sq  = sum(t ** 2 for t in throughputs)
        tp_arr_sum = sum(throughputs)
        jains = (tp_arr_sum ** 2) / (n_intersections * max(tp_arr_sq, 1))
        fairness_reward = jains  # in [0, 1]

    # --- Compose reward ---
    # Normalise by n_intersections where applicable
    reward.throughput_bonus  = W_THROUGHPUT  * (total_tp / n_intersections)
    reward.queue_penalty     = W_QUEUE       * (total_queue / n_lanes)
    reward.wait_penalty      = W_WAIT        * (total_wait / n_intersections)
    reward.switch_penalty    = W_SWITCH      * (switch_penalty / n_intersections)
    reward.emergency_bonus   = W_EMERGENCY   * min(emergency_reward, MAX_EMERGENCY_REWARD)
    reward.spillback_penalty = W_SPILLBACK   * (total_spillback / n_intersections)
    reward.starvation_penalty = W_STARVATION * (total_starvation / n_lanes)
    reward.fairness_bonus    = W_FAIRNESS    * fairness_reward

    return reward
