"""Bounded [0,1] per-step reward for TrafficSignalEnv.

Formula:  reward = Σ (weight_i × quality_i)  ∈ [0, 1]

All quality components are independently normalized to [0, 1].
Weights sum to exactly 1.0, which mathematically guarantees the total
reward is in [0, 1] — a safety clip is added as belt-and-braces.

Component        Weight  Meaning (1.0 = perfect, 0.0 = worst)
---------------  ------  ----------------------------------------
throughput        0.40   vehicles discharged this step / max possible
queue             0.25   1 − worst-lane queue fraction
direction         0.13   serving the direction with the bigger queue
emergency         0.10   emergency lane in green (or no emergency)
spillback         0.05   1 − fraction of intersections with spillback
smooth            0.04   1 − fraction of intersections that switched
starvation        0.02   1 − worst-lane consecutive-red-steps fraction
wait              0.01   1 − normalised vehicle-queue pressure
fairness          0.00   (disabled — absorbed into direction)
                 ------
                  1.00
"""
from __future__ import annotations

from typing import Dict, List

from env.schemas import (
    EmergencyType,
    PhaseEnum,
    TrafficReward,
    TrafficState,
)
from env.simulator import PHASE_GREEN_LANES

# ---------------------------------------------------------------------------
# Normalization references
# ---------------------------------------------------------------------------
_MAX_TP_PER_INTER  = 6.0    # 2 green lanes × 3 discharge_rate
_MAX_QUEUE_STEPS   = 15     # consecutive red steps before lane is "starved"
_NS_LANES = [0, 1]
_EW_LANES = [2, 3]

# Emergency: score when the vehicle is NOT served (lower = more urgent)
_UNSERVED_SCORE = {
    EmergencyType.POLICE:    0.5,
    EmergencyType.FIRE:      0.2,
    EmergencyType.AMBULANCE: 0.0,
}

# ---------------------------------------------------------------------------
# Weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
#  Throughput and queue are the primary grader signals (keep dominant).
#  Direction bonus rewards correct phase choice for the queue imbalance.
#  Emergency is critical in hard mode.
#  Spillback matters for medium/hard (propagation).
#  Smooth (switch penalty) raised so oscillation is meaningfully penalized.
#  Starvation raised so holding one phase forever is punished.
#  Wait is de-emphasized (largely redundant with queue).
#  Fairness weight is minimal.
# ---------------------------------------------------------------------------
W_THROUGHPUT = 0.40   # primary grader target
W_QUEUE      = 0.25   # queue pressure matters
W_DIRECTION  = 0.13   # direction alignment bonus
W_EMERGENCY  = 0.10   # emergency (active in hard)
W_SPILLBACK  = 0.05   # spillback propagation
W_SMOOTH     = 0.03   # switch penalty
W_ALL_RED    = 0.02   # penalize useless ALL_RED (no yellow, no emergency)
W_STARVATION = 0.01   # starvation penalty
W_WAIT       = 0.01   # queue pressure (de-emphasized)
# sum: 0.40+0.25+0.13+0.10+0.05+0.03+0.02+0.01+0.01 = 1.00


def compute_reward(
    state: TrafficState,
    prev_state: TrafficState | None,
    step_stats: dict,
    max_queue: int,
) -> TrafficReward:
    """Return a TrafficReward whose .total is always in [0.0, 1.0].

    Uses per-step LaneState.throughput (reset each step by the simulator),
    NOT the cumulative IntersectionState.total_throughput, so the signal
    is stable across all steps of an episode.
    """
    n = len(state.intersections)
    if n == 0:
        return TrafficReward()

    stat_by_id: Dict[int, dict] = {
        s["intersection_id"]: s
        for s in step_stats.get("intersections", [])
    }

    # ── Accumulators ────────────────────────────────────────────────────
    sum_tp_frac    = 0.0
    max_q_frac     = 0.0
    sum_q_pressure = 0.0
    n_spill        = 0
    worst_starv    = 0
    dir_hits       = 0.0
    em_scores: List[float] = []
    tp_per_inter:  List[float] = []
    n_switched     = 0
    n_all_red      = 0   # intersections in useless ALL_RED (no yellow, no emergency)

    for istat in state.intersections:
        iid  = istat.intersection_id
        stat = stat_by_id.get(iid, {})

        # ── Per-step throughput (from LaneState, reset each step) ───────
        tp_step = float(sum(l.throughput for l in istat.lanes))
        tp_frac = min(1.0, tp_step / _MAX_TP_PER_INTER)
        sum_tp_frac += tp_frac
        tp_per_inter.append(tp_step)

        # ── Queue pressure ──────────────────────────────────────────────
        for lane in istat.lanes:
            frac = lane.queue_length / max(max_queue, 1)
            if frac > max_q_frac:
                max_q_frac = frac
            sum_q_pressure += lane.queue_length

        # ── Spillback (per-step boolean from step_stats) ────────────────
        if stat.get("spillback", False):
            n_spill += 1

        # ── Starvation (consecutive red steps, worst lane) ──────────────
        for lane_stat in stat.get("lanes", []):
            s = int(lane_stat.get("starvation", 0))
            if s > worst_starv:
                worst_starv = s

        # ── Emergency quality ───────────────────────────────────────────
        if istat.emergency_active == EmergencyType.NONE:
            em_scores.append(1.0)
        else:
            green = PHASE_GREEN_LANES.get(istat.current_phase, [])
            if istat.emergency_lane in green:
                em_scores.append(1.0)
            else:
                em_scores.append(_UNSERVED_SCORE.get(istat.emergency_active, 0.2))

        # ── Direction alignment ─────────────────────────────────────────
        ns_q = sum(l.queue_length for l in istat.lanes if l.lane_id in _NS_LANES)
        ew_q = sum(l.queue_length for l in istat.lanes if l.lane_id in _EW_LANES)
        ph   = istat.current_phase
        if ns_q > ew_q and ph == PhaseEnum.NS_GREEN:
            dir_hits += 1.0
        elif ew_q > ns_q and ph == PhaseEnum.EW_GREEN:
            dir_hits += 1.0
        elif ns_q == ew_q:
            dir_hits += 0.5

        # ── Switch count ────────────────────────────────────────────────
        if stat.get("phase_switched", False):
            n_switched += 1

        # ── ALL_RED abuse detection ─────────────────────────────────
        # Penalize ALL_RED when there is no yellow and no emergency
        if (istat.current_phase == PhaseEnum.ALL_RED
                and istat.yellow_remaining == 0
                and istat.emergency_active.value == 0):
            n_all_red += 1

    # ── Fairness: Jain's index on per-step throughput ───────────────────
    if n > 1 and max(tp_per_inter) > 0:
        s  = sum(tp_per_inter)
        sq = sum(t * t for t in tp_per_inter)
        fair_q = min(1.0, (s * s) / max(n * sq, 1e-9))
    else:
        fair_q = 1.0

    # ── Compose normalized quality scores [0, 1] ────────────────────────
    n_lanes       = n * 4
    q_throughput  = sum_tp_frac / n
    q_queue       = 1.0 - max_q_frac
    q_direction   = dir_hits / n
    q_emergency   = sum(em_scores) / len(em_scores)
    q_spillback   = 1.0 - min(1.0, n_spill / n)
    q_wait        = 1.0 - min(1.0, sum_q_pressure / max(n_lanes * max_queue, 1))
    q_starvation  = 1.0 - min(1.0, worst_starv / _MAX_QUEUE_STEPS)
    q_smooth      = 1.0 - (n_switched / n)
    q_all_red     = 1.0 - min(1.0, n_all_red / n)   # 1.0 = no useless ALL_RED

    # ── Weighted sum → guaranteed in [0, 1] ─────────────────────────────
    reward_value = (
        W_THROUGHPUT * q_throughput
        + W_QUEUE      * q_queue
        + W_DIRECTION  * q_direction
        + W_EMERGENCY  * q_emergency
        + W_SPILLBACK  * q_spillback
        + W_WAIT       * q_wait
        + W_STARVATION * q_starvation
        + W_SMOOTH     * q_smooth
        + W_ALL_RED    * q_all_red
    )
    reward_value = max(0.0, min(1.0, reward_value))

    # ── Store in TrafficReward fields so .total == reward_value ─────────
    # Each field holds its weighted contribution (all positive, sum = reward_value).
    # Field names retain original semantics in to_dict() for debugging.
    reward = TrafficReward()
    reward.throughput_bonus    = W_THROUGHPUT * q_throughput
    reward.queue_penalty       = W_QUEUE      * q_queue
    reward.wait_penalty        = W_WAIT       * q_wait
    reward.switch_penalty      = W_SMOOTH     * q_smooth + W_ALL_RED * q_all_red
    reward.emergency_bonus     = W_EMERGENCY  * q_emergency
    reward.spillback_penalty   = W_SPILLBACK  * q_spillback
    reward.starvation_penalty  = W_STARVATION * q_starvation
    reward.fairness_bonus      = W_DIRECTION  * q_direction   # direction aligned here

    return reward
