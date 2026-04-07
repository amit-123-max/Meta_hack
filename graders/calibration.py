"""Calibration utility — data-driven normalization bounds for graders.

Computes robust (winsorized, trimmed) percentile ranges from trajectory data.
Graders use these bounds for calibrated normalization instead of static defaults.

Tracked keys
------------
  tp          : per-step global throughput (vehicles/step)
  queue       : worst per-step queue across all lanes
  spillback   : fraction of intersections spilling per step
  switch      : overall phase switch rate (switches / (steps * n_inters))
  wait        : per-step avg queue length (proxy for wait)
  fairness    : episode-level Jain's index
  all_red_rate: fraction of steps where ALL_RED was chosen
  starvation  : fraction of intersections that were starved per episode

Usage
-----
    from graders.calibration import compute_calibration, validate_calibration
    calibration = compute_calibration(trajectories)
    if not validate_calibration(calibration):
        print("Calibration incomplete — graders will use static defaults")
    grader = EasyGrader(calibration=calibration)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _winsorize(
    data: List[float], lo_pct: float = 5.0, hi_pct: float = 95.0
) -> List[float]:
    """Clip values to [lo_pct, hi_pct] percentile before further analysis."""
    if len(data) < 4:
        return data
    lo = float(np.percentile(data, lo_pct))
    hi = float(np.percentile(data, hi_pct))
    return [max(lo, min(hi, v)) for v in data]


def _trimmed_mean(
    data: List[float], trim_pct: float = 0.10
) -> float:
    """Mean after trimming the bottom and top trim_pct fraction of values."""
    if not data:
        return 0.0
    n = len(data)
    k = max(0, int(n * trim_pct))
    sorted_data = sorted(data)
    trimmed = sorted_data[k: n - k] if k > 0 else sorted_data
    return float(np.mean(trimmed)) if trimmed else float(np.mean(sorted_data))


def _percentile_bounds(
    data: List[float], lo_pct: float = 10.0, hi_pct: float = 90.0
) -> Tuple[float, float]:
    """Return (p_lo, p_hi) percentile pair from data."""
    if not data:
        return (0.0, 1.0)
    arr = np.array(data, dtype=float)
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return (lo, hi)


# ---------------------------------------------------------------------------
# Primary calibration function
# ---------------------------------------------------------------------------

def compute_calibration(
    trajectories: List[List[Dict]],
    lo_pct: float = 10.0,
    hi_pct: float = 90.0,
    winsorize_outer: float = 5.0,
) -> Dict[str, Tuple[float, float]]:
    """Compute calibration bounds from a collection of episode trajectories.

    Args:
        trajectories   : list of episode trajectories (each a list of step dicts).
        lo_pct         : lower percentile for normalization lower bound.
        hi_pct         : upper percentile for normalization upper bound.
        winsorize_outer: outer clip percentile applied before bound computation.

    Returns:
        dict mapping metric key → (lo_bound, hi_bound)
    """
    tp_vals: List[float] = []
    queue_vals: List[float] = []
    spillback_vals: List[float] = []
    switch_vals: List[float] = []
    wait_vals: List[float] = []

    # Episode-level metrics (one value per trajectory)
    fairness_vals: List[float] = []
    all_red_rate_vals: List[float] = []
    starvation_vals: List[float] = []

    for traj in trajectories:
        if not traj:
            continue

        n_steps = len(traj)
        snap_0 = traj[0].get("state_snapshot", {})
        n_inters = max(len(snap_0.get("intersections", [])), 1)

        # Cumulative phase switches for the episode
        switches_in_traj = int(
            traj[-1].get("state_snapshot", {}).get("phase_switches", 0)
        )
        switch_vals.append(switches_in_traj / max(n_steps * n_inters, 1))

        # Episode-level all_red_rate
        all_red_steps = 0
        inter_totals: Dict[int, float] = {}
        expected_per_inter = max(n_steps * 0.5, 1.0)

        for step_data in traj:
            snap = step_data.get("state_snapshot", {})
            tp_vals.append(float(snap.get("global_throughput", 0.0)))
            wait_vals.append(float(snap.get("global_avg_wait", 0.0)))

            intersections = snap.get("intersections", [])
            n_inter_step = max(len(intersections), 1)

            step_worst_q = 0.0
            step_spills = 0

            for inter in intersections:
                iid = inter.get("id", 0)
                itp = float(inter.get("throughput", 0.0))
                inter_totals[iid] = inter_totals.get(iid, 0.0) + itp

                qs = inter.get("queues", [])
                if qs:
                    local_worst = max(float(q) for q in qs)
                    step_worst_q = max(step_worst_q, local_worst)

                if inter.get("spillback", 0) > 0:
                    step_spills += 1

                # ALL_RED tracking
                if inter.get("phase", -1) == 2:   # PhaseEnum.ALL_RED == 2
                    all_red_steps += 1

            queue_vals.append(step_worst_q)
            spillback_vals.append(step_spills / n_inter_step)

        # Episode-level all_red_rate
        all_red_rate_vals.append(all_red_steps / max(n_steps * n_inters, 1))

        # Episode-level Jain's fairness
        if len(inter_totals) > 1:
            vals = list(inter_totals.values())
            s = sum(vals)
            sq = sum(v * v for v in vals)
            n_i = len(vals)
            jain = float(min(1.0, (s * s) / (n_i * sq))) if sq > 0 else 1.0
        else:
            jain = 1.0
        fairness_vals.append(jain)

        # Episode-level starvation rate
        starved = sum(
            1 for v in inter_totals.values()
            if v < expected_per_inter * 0.02
        )
        starvation_vals.append(starved / max(n_inters, 1))

    # Build calibration dict with winsorization then percentile bounds
    kalib: Dict[str, Tuple[float, float]] = {}

    def _add(key: str, data: List[float]) -> None:
        if data:
            ws = _winsorize(data, winsorize_outer, 100.0 - winsorize_outer)
            kalib[key] = _percentile_bounds(ws, lo_pct, hi_pct)

    _add("tp", tp_vals)
    _add("queue", queue_vals)
    _add("spillback", spillback_vals)
    _add("switch", switch_vals)
    _add("wait", wait_vals)
    _add("fairness", fairness_vals)
    _add("all_red_rate", all_red_rate_vals)
    _add("starvation", starvation_vals)

    return kalib


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"tp", "queue", "spillback", "switch"}
OPTIONAL_KEYS = {"wait", "fairness", "all_red_rate", "starvation"}


def validate_calibration(
    calibration: Optional[Dict[str, Tuple[float, float]]]
) -> bool:
    """Return True if calibration contains all required keys."""
    if not calibration:
        return False
    return all(k in calibration for k in REQUIRED_KEYS)


def calibration_summary(
    calibration: Dict[str, Tuple[float, float]]
) -> str:
    """Return a human-readable summary of calibration bounds."""
    lines = ["Calibration bounds:"]
    for k in sorted(calibration):
        lo, hi = calibration[k]
        lines.append(f"  {k:<16} lo={lo:.4f}  hi={hi:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    print("Calibration utility — example usage:")
    print("  from graders.calibration import compute_calibration")
    print("  calib = compute_calibration(list_of_trajectories)")
    print()

    # Demo with synthetic data
    rng = np.random.default_rng(0)
    dummy_traj = []
    for _ in range(5):
        ep = []
        for step_i in range(20):
            ep.append({
                "state_snapshot": {
                    "global_throughput": float(rng.poisson(3.0)),
                    "global_avg_wait": float(rng.uniform(0, 5)),
                    "phase_switches": step_i // 4,
                    "intersections": [
                        {
                            "id": j,
                            "queues": list(rng.integers(0, 8, size=4).astype(float)),
                            "spillback": int(rng.random() < 0.1),
                            "throughput": float(rng.poisson(1.5)),
                            "phase": int(rng.integers(0, 3)),
                        }
                        for j in range(4)
                    ],
                }
            })
        dummy_traj.append(ep)

    calib = compute_calibration(dummy_traj)
    print(calibration_summary(calib))
    print(f"\nValid: {validate_calibration(calib)}")
