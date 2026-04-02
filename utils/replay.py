"""Episode replay and analytics — produces per-episode summaries."""
from __future__ import annotations

import json
import statistics
from typing import Any, Dict, List, Optional


def build_analytics(trajectory: List[Dict[str, Any]], task_id: str) -> Dict:
    """Build analytics dict from episode trajectory.

    Returns:
        dict with keys:
          task_id, n_steps, total_reward,
          avg_wait_per_lane, emergency_delay_mean,
          phase_timeline, queue_heatmap,
          violations_count, throughput_summary
    """
    if not trajectory:
        return {}

    n_steps        = len(trajectory)
    total_reward   = sum(s.get("reward", 0.0) for s in trajectory)
    rewards        = [s.get("reward", 0.0) for s in trajectory]

    # Per-step averages
    avg_waits: List[float] = []
    throughputs:  List[int] = []
    all_queues:   List[float] = []
    phase_timeline: List[Dict] = []
    violations     = 0

    # Emergency delays (from info or trajectory)
    emergency_delays: List[float] = []

    for step_data in trajectory:
        snap = step_data.get("state_snapshot", {})
        avg_waits.append(float(snap.get("global_avg_wait", 0.0)))
        throughputs.append(int(snap.get("global_throughput", 0)))

        inter_phases = []
        for inter in snap.get("intersections", []):
            queues = inter.get("queues", [])
            if queues:
                all_queues.append(statistics.mean(q for q in queues if q >= 0))
            inter_phases.append({
                "id":    inter.get("id", 0),
                "phase": inter.get("phase", -1),
                "emergency": inter.get("emergency", 0),
            })
            # Count violations (emergency present + phase ≠ emergency lane phase)
            em = int(inter.get("emergency", 0))
            if em > 0 and inter.get("phase", 2) == 2:
                violations += 1

        phase_timeline.append({
            "step":         snap.get("step", 0),
            "intersections": inter_phases,
        })

    # Emergency delays from reward breakdown
    em_delays_found = set()
    for step_data in trajectory:
        rb = step_data.get("reward_breakdown", {})
        eb = rb.get("emergency_bonus", 0.0)
        if eb > 0.0:
            eb_delay = max(0.0, (2.0 - eb) * 5.0)   # rough estimate
            em_delays_found.add(rb.get("emergency_bonus", 0.0))

    emergency_delay_mean = statistics.mean(emergency_delays) if emergency_delays else None

    # Queue heatmap — average queue per intersection per time bucket
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
            "bucket": bucket_idx // bucket_size,
            "start_step": bucket_idx,
            "intersections": {
                str(iid): round(statistics.mean(vals), 2)
                for iid, vals in bucket_queues.items()
            },
        }
        queue_heatmap.append(hm_row)

    # Throughput summary
    throughput_summary = {
        "total":   sum(throughputs),
        "mean":    round(statistics.mean(throughputs) if throughputs else 0.0, 2),
        "max":     max(throughputs) if throughputs else 0,
        "min":     min(throughputs) if throughputs else 0,
    }

    return {
        "task_id":             task_id,
        "n_steps":             n_steps,
        "total_reward":        round(total_reward, 4),
        "avg_wait_per_lane":   round(statistics.mean(avg_waits) if avg_waits else 0.0, 4),
        "avg_queue":           round(statistics.mean(all_queues) if all_queues else 0.0, 4),
        "emergency_delay_mean": emergency_delay_mean,
        "violations_count":    violations,
        "throughput_summary":  throughput_summary,
        "phase_timeline":      phase_timeline,
        "queue_heatmap":       queue_heatmap,
        "reward_per_step": {
            "mean": round(statistics.mean(rewards), 6),
            "min":  round(min(rewards), 6),
            "max":  round(max(rewards), 6),
        },
    }


def print_analytics(analytics: Dict, indent: int = 2) -> None:
    """Pretty-print analytics dict to stdout."""
    # Remove phase_timeline and heatmap from terminal output (too verbose)
    display = {k: v for k, v in analytics.items()
               if k not in ("phase_timeline", "queue_heatmap")}
    print(json.dumps(display, indent=indent, default=str))
