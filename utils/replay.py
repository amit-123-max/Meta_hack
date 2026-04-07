"""Episode replay and analytics — produces per-episode summaries.

v2 improvements
---------------
- Emergency delay uses REAL simulator event log (not reward bonus estimate).
- Violations are detailed string descriptions, not just a count.
- Fairness summary added (Jain's index, episode-level).
- Starvation summary added.
- best_step / worst_step identified.
- EpisodeFeedback surfaces as part of analytics when available.
"""
from __future__ import annotations

import json
import statistics
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Main analytics builder
# ---------------------------------------------------------------------------

def build_analytics(
    trajectory: List[Dict[str, Any]],
    task_id: str,
) -> Dict:
    """Build analytics dict from episode trajectory.

    Returns:
        dict with keys:
          task_id, n_steps, total_reward,
          avg_wait_per_lane, emergency_delay_mean, emergency_events,
          phase_timeline, queue_heatmap,
          violations_detail, violations_count,
          throughput_summary, fairness_summary, starvation_summary,
          best_step, worst_step, reward_per_step
    """
    if not trajectory:
        return {}

    n_steps        = len(trajectory)
    total_reward   = sum(s.get("reward", 0.0) for s in trajectory)
    rewards        = [s.get("reward", 0.0) for s in trajectory]

    # ------------------------------------------------------------------
    # Per-step metrics
    # ------------------------------------------------------------------
    avg_waits:    List[float] = []
    throughputs:  List[int]   = []
    all_queues:   List[float] = []
    phase_timeline: List[Dict] = []
    violations_detail: List[str] = []

    # Emergency delays — use explicit event log from trajectory snapshot
    emergency_events: List[Dict] = []
    seen_em_keys = set()

    # Episode-level intersection throughput for fairness
    inter_totals: Dict[int, float] = {}

    # ALL_RED and phase counters
    all_red_steps = 0
    total_inter_steps = 0

    for step_data in trajectory:
        snap = step_data.get("state_snapshot", {})
        r    = step_data.get("reward", 0.0)
        avg_waits.append(float(snap.get("global_avg_wait", 0.0)))
        throughputs.append(int(snap.get("global_throughput", 0)))

        inter_phases: List[Dict] = []
        for inter in snap.get("intersections", []):
            iid    = inter.get("id", 0)
            phase  = inter.get("phase", -1)
            queues = inter.get("queues", [])
            em     = int(inter.get("emergency", 0))

            if queues:
                all_queues.append(statistics.mean(q for q in queues if q >= 0))

            inter_totals[iid] = inter_totals.get(iid, 0.0) + float(inter.get("throughput", 0.0))
            inter_phases.append({
                "id":        iid,
                "phase":     phase,
                "emergency": em,
            })

            if phase == 2:  # ALL_RED
                all_red_steps += 1

            # Violation: emergency present but ALL_RED chosen (not serving)
            if em > 0 and phase == 2:
                violations_detail.append(
                    f"step={snap.get('step','?')} I{iid}: emergency={em} but phase=ALL_RED"
                )

            total_inter_steps += 1

        phase_timeline.append({
            "step":          snap.get("step", 0),
            "intersections": inter_phases,
        })

        # Pull real emergency events from the snapshot (accumulates across steps)
        for ev in snap.get("emergency_events", []):
            ev_key = (ev.get("iid"), ev.get("lane_id"), ev.get("arrival_step"))
            if ev_key not in seen_em_keys:
                seen_em_keys.add(ev_key)
                emergency_events.append(ev)

    # ------------------------------------------------------------------
    # Emergency delay statistics (REAL delays from simulator events)
    # ------------------------------------------------------------------
    served_delays   = [ev["latency_steps"] for ev in emergency_events if ev.get("served")]
    unserved_events = [ev for ev in emergency_events if not ev.get("served")]
    emergency_delay_mean = (
        round(statistics.mean(served_delays), 3) if served_delays else None
    )

    if unserved_events:
        for ev in unserved_events:
            violations_detail.append(
                f"Emergency NEGLECTED: type={ev.get('etype','?')} at "
                f"I{ev.get('iid','?')} lane={ev.get('lane_id','?')} "
                f"latency={ev.get('latency_steps','?')}s"
            )

    # ------------------------------------------------------------------
    # Fairness (Jain's index, episode-level)
    # ------------------------------------------------------------------
    fairness_summary: Dict[str, Any] = {"n_intersections": len(inter_totals)}
    if len(inter_totals) > 1:
        vals = list(inter_totals.values())
        s  = sum(vals)
        sq = sum(v * v for v in vals)
        n  = len(vals)
        jain = float(min(1.0, (s * s) / (n * sq))) if sq > 0 else 1.0
        fairness_summary["jain_index"]  = round(jain, 4)
        fairness_summary["per_inter"]   = {str(k): round(v, 2) for k, v in inter_totals.items()}
        fairness_summary["min_throughput"] = round(min(vals), 2)
        fairness_summary["max_throughput"] = round(max(vals), 2)
    else:
        fairness_summary["jain_index"] = 1.0

    # ------------------------------------------------------------------
    # Starvation summary
    # ------------------------------------------------------------------
    expected_per_inter = max(n_steps * 0.5, 1.0)
    starvation_summary: Dict[str, Any] = {
        "expected_per_inter_min": round(expected_per_inter * 0.02, 2),
        "starved": [
            {"intersection": k, "total_throughput": round(v, 2)}
            for k, v in inter_totals.items()
            if v < expected_per_inter * 0.02
        ],
    }

    # ------------------------------------------------------------------
    # ALL_RED rate
    # ------------------------------------------------------------------
    all_red_rate = all_red_steps / max(total_inter_steps, 1)
    if all_red_rate > 0.30:
        violations_detail.append(
            f"ALL_RED abuse: {all_red_rate:.1%} of intersection-steps in ALL_RED"
        )

    # ------------------------------------------------------------------
    # Best / worst step
    # ------------------------------------------------------------------
    best_step  = int(rewards.index(max(rewards))) + 1 if rewards else 0
    worst_step = int(rewards.index(min(rewards))) + 1 if rewards else 0

    # ------------------------------------------------------------------
    # Queue heatmap (10 time buckets)
    # ------------------------------------------------------------------
    bucket_size = max(1, n_steps // 10)
    queue_heatmap: List[Dict] = []
    for bucket_idx in range(0, n_steps, bucket_size):
        bucket = trajectory[bucket_idx: bucket_idx + bucket_size]
        bucket_queues: Dict[int, List[float]] = {}
        for step_data in bucket:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                iid = inter.get("id", 0)
                qs  = inter.get("queues", [])
                if qs:
                    bucket_queues.setdefault(iid, []).append(statistics.mean(qs))
        hm_row = {
            "bucket":      bucket_idx // bucket_size,
            "start_step":  bucket_idx,
            "intersections": {
                str(iid): round(statistics.mean(vals), 2)
                for iid, vals in bucket_queues.items()
            },
        }
        queue_heatmap.append(hm_row)

    # ------------------------------------------------------------------
    # Throughput summary
    # ------------------------------------------------------------------
    throughput_summary = {
        "total":  sum(throughputs),
        "mean":   round(statistics.mean(throughputs) if throughputs else 0.0, 2),
        "max":    max(throughputs) if throughputs else 0,
        "min":    min(throughputs) if throughputs else 0,
    }

    return {
        "task_id":              task_id,
        "n_steps":              n_steps,
        "total_reward":         round(total_reward, 4),
        "avg_wait_per_lane":    round(statistics.mean(avg_waits) if avg_waits else 0.0, 4),
        "avg_queue":            round(statistics.mean(all_queues) if all_queues else 0.0, 4),
        "emergency_delay_mean": emergency_delay_mean,
        "emergency_events":     emergency_events,
        "violations_detail":    violations_detail,
        "violations_count":     len(violations_detail),
        "fairness_summary":     fairness_summary,
        "starvation_summary":   starvation_summary,
        "all_red_rate":         round(all_red_rate, 4),
        "best_step":            best_step,
        "worst_step":           worst_step,
        "throughput_summary":   throughput_summary,
        "phase_timeline":       phase_timeline,
        "queue_heatmap":        queue_heatmap,
        "reward_per_step": {
            "mean": round(statistics.mean(rewards), 6),
            "min":  round(min(rewards), 6),
            "max":  round(max(rewards), 6),
        },
    }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_analytics(analytics: Dict, indent: int = 2) -> None:
    """Pretty-print analytics dict to stdout (excluding verbose lists)."""
    skip = {"phase_timeline", "queue_heatmap", "emergency_events"}
    display = {k: v for k, v in analytics.items() if k not in skip}
    print(json.dumps(display, indent=indent, default=str))
