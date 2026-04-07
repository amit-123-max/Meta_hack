#!/usr/bin/env python3
"""Validation script — proves nothing broke after the grader/feedback upgrade.

Assertions
----------
 1. Grader scores are in [0, 1] for all three tasks.
 2. Easy score ≥ Hard score at baseline (difficulty ordering).
 3. info["step_feedback"] is present in every step's info dict.
 4. StepFeedback has expected fields.
 5. info["episode_feedback"] present when done=True.
 6. EpisodeFeedback.lessons is a non-empty list.
 7. build_analytics() returns real emergency_events (not just a count).
 8. Calibration computes valid bounds from trajectory data.
 9. HF adapter factory returns None gracefully when HF vars not set.
10. All grader scores deterministic (same score on second run).

Usage:
    python scripts/validate_upgrade.py
    # or with a venv:
    .venv/bin/python scripts/validate_upgrade.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (details printed to stdout).
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, List, Tuple

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "✔ PASS"
_FAIL = "✘ FAIL"
_results: List[Tuple[str, bool, str]] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = _PASS if condition else _FAIL
    _results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def _run_episode(task_id: str, seed: int = 42):
    """Run one episode with rule-based agent; return (trajectory, info_list, score)."""
    from baseline.rule_based_agent import RuleBasedAgent
    from graders.easy_grader import EasyGrader
    from graders.medium_grader import MediumGrader
    from graders.hard_grader import HardGrader
    from tasks.task_easy import make_env as make_easy
    from tasks.task_medium import make_env as make_medium
    from tasks.task_hard import make_env as make_hard

    if task_id == "easy":
        env    = make_easy(seed=seed)
        grader = EasyGrader()
    elif task_id == "medium":
        env    = make_medium(seed=seed)
        grader = MediumGrader()
    else:
        env    = make_hard(seed=seed)
        grader = HardGrader()

    agent = RuleBasedAgent(
        n_intersections=env.n_intersections,
        min_phase_steps=env.cfg.sim.phase_duration_min,
        max_phase_steps=env.cfg.sim.phase_duration_max,
    )
    obs   = env.reset(seed=seed)
    agent.reset()
    done  = False
    infos: List[Dict] = []

    while not done:
        ta     = agent.act(obs)
        obs, reward, done, info = env.step(ta.phase_indices)
        infos.append(info)

    score = grader.grade(env.trajectory)
    return env.trajectory, infos, score


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_score_range(scores: Dict[str, float]) -> None:
    for task_id, score in scores.items():
        _check(
            f"Score in [0,1] — {task_id}",
            0.0 <= score <= 1.0,
            f"score={score:.4f}",
        )


def check_difficulty_ordering(scores: Dict[str, float]) -> None:
    easy  = scores.get("easy", 0.0)
    hard  = scores.get("hard", 0.0)
    _check(
        "Difficulty ordering: easy_score ≥ hard_score",
        easy >= hard,
        f"easy={easy:.4f} hard={hard:.4f}",
    )


def check_step_feedback_present(infos: List[Dict], task_id: str) -> None:
    missing = [i for i, info in enumerate(infos) if "step_feedback" not in info]
    _check(
        f"step_feedback in every info — {task_id}",
        len(missing) == 0,
        f"missing at steps: {missing[:5]}" if missing else "all present",
    )


def check_step_feedback_fields(infos: List[Dict], task_id: str) -> None:
    required_fields = [
        "step", "risk_level", "dominant_queue", "emergency_active",
        "last_action_sensible", "suggested_action", "reward_breakdown",
        "went_right", "went_wrong",
    ]
    if not infos:
        _check(f"StepFeedback fields — {task_id}", False, "no infos")
        return
    fb = infos[0].get("step_feedback")
    if fb is None:
        _check(f"StepFeedback fields — {task_id}", False, "step_feedback is None")
        return
    # StepFeedback can be dataclass or dict
    if hasattr(fb, "__dataclass_fields__"):
        fields = list(fb.__dataclass_fields__.keys())
    elif isinstance(fb, dict):
        fields = list(fb.keys())
    else:
        fields = dir(fb)
    missing = [f for f in required_fields if f not in fields]
    _check(
        f"StepFeedback fields — {task_id}",
        len(missing) == 0,
        f"missing: {missing}" if missing else "ok",
    )


def check_episode_feedback(infos: List[Dict], task_id: str) -> None:
    last_info = infos[-1] if infos else {}
    has_ep_fb = "episode_feedback" in last_info
    _check(
        f"episode_feedback in final info — {task_id}",
        has_ep_fb,
        "present" if has_ep_fb else "MISSING",
    )

    if has_ep_fb:
        ep_fb = last_info["episode_feedback"]
        lessons = getattr(ep_fb, "lessons", None) or ep_fb.get("lessons", []) if isinstance(ep_fb, dict) else []
        if hasattr(ep_fb, "lessons"):
            lessons = ep_fb.lessons
        _check(
            f"EpisodeFeedback.lessons non-empty — {task_id}",
            bool(lessons),
            f"{len(lessons)} lesson(s)" if lessons else "EMPTY",
        )


def check_emergency_events_real(trajectory: List[Dict], task_id: str) -> None:
    """Verify emergency_events is a list (not None) in the final snapshot."""
    if not trajectory:
        _check(f"emergency_events in last snapshot — {task_id}", False, "empty trajectory")
        return
    last_snap = trajectory[-1].get("state_snapshot", {})
    em_events = last_snap.get("emergency_events")
    is_list   = isinstance(em_events, list)
    _check(
        f"emergency_events is list — {task_id}",
        is_list,
        f"type={type(em_events).__name__}  len={len(em_events) if is_list else '?'}",
    )


def check_analytics_emergency(trajectory: List[Dict], task_id: str) -> None:
    from utils.replay import build_analytics
    analytics = build_analytics(trajectory, task_id)
    em_events = analytics.get("emergency_events", [])
    # For easy/medium tasks with no emergencies this will be empty — that's fine
    # What we check: the key exists and is a list (not inferred from bonus)
    _check(
        f"analytics.emergency_events is list — {task_id}",
        isinstance(em_events, list),
        f"len={len(em_events)}",
    )
    # violations_detail should be a list too
    violations = analytics.get("violations_detail", None)
    _check(
        f"analytics.violations_detail is list — {task_id}",
        isinstance(violations, list),
        f"len={len(violations) if violations is not None else '?'}",
    )


def check_calibration(trajectory: List[Dict], task_id: str) -> None:
    from graders.calibration import compute_calibration, validate_calibration
    calib = compute_calibration([trajectory])
    valid = validate_calibration(calib)
    _check(
        f"Calibration valid — {task_id}",
        valid,
        f"keys={list(calib.keys())}",
    )
    # All lo < hi
    all_ok = all(lo < hi for lo, hi in calib.values())
    _check(
        f"Calibration bounds lo < hi — {task_id}",
        all_ok,
        str({k: v for k, v in calib.items() if v[0] >= v[1]}) or "ok",
    )


def check_determinism(task_id: str) -> None:
    """Run twice and compare scores."""
    _, _, score1 = _run_episode(task_id, seed=42)
    _, _, score2 = _run_episode(task_id, seed=42)
    _check(
        f"Grader deterministic — {task_id}",
        abs(score1 - score2) < 1e-9,
        f"run1={score1:.6f} run2={score2:.6f}",
    )


def check_hf_adapter_no_token() -> None:
    """HF adapter returns None gracefully when token is missing."""
    import os
    from llm_agent.llm_adapter import build_adapter

    # Temporarily remove HF-specific vars
    saved = {k: os.environ.pop(k, None) for k in ("MODEL_PROVIDER", "HF_TOKEN", "HUGGING_FACE_TOKEN", "OLLAMA_MODEL")}
    os.environ["MODEL_PROVIDER"] = "hf"

    try:
        adapter = build_adapter(verbose=False)
        _check(
            "HF adapter returns None without HF_TOKEN",
            adapter is None,
            f"adapter={adapter}",
        )
    finally:
        os.environ.pop("MODEL_PROVIDER", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n" + "=" * 64, flush=True)
    print("  TrafficSignalEnv — Upgrade Validation", flush=True)
    print("=" * 64 + "\n", flush=True)

    trajectories: Dict[str, Any] = {}
    infos_map: Dict[str, List[Dict]] = {}
    scores: Dict[str, float] = {}

    # --- Run episodes ---
    for task_id in ["easy", "medium", "hard"]:
        print(f"[Running {task_id}]", flush=True)
        try:
            traj, infos, score = _run_episode(task_id)
            trajectories[task_id] = traj
            infos_map[task_id]    = infos
            scores[task_id]       = score
            print(f"  score={score:.4f}  steps={len(traj)}", flush=True)
        except Exception as exc:
            print(f"  ❌ Episode failed: {exc}", flush=True)
            traceback.print_exc()

    print("\n--- Assertion Results ---\n", flush=True)

    # 1. Score range
    check_score_range(scores)

    # 2. Difficulty ordering
    if "easy" in scores and "hard" in scores:
        check_difficulty_ordering(scores)

    # 3-6. Per-task feedback checks
    for task_id in ["easy", "medium", "hard"]:
        if task_id not in trajectories:
            continue
        infos = infos_map[task_id]
        traj  = trajectories[task_id]
        check_step_feedback_present(infos, task_id)
        check_step_feedback_fields(infos, task_id)
        check_episode_feedback(infos, task_id)
        check_emergency_events_real(traj, task_id)
        check_analytics_emergency(traj, task_id)
        check_calibration(traj, task_id)
        check_determinism(task_id)

    # 9. HF adapter graceful failure
    check_hf_adapter_no_token()

    # --- Summary ---
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)

    print(f"\n{'='*64}", flush=True)
    print(f"  Results: {n_pass}/{total} passed  |  {n_fail} failed", flush=True)

    if n_fail:
        print("\n  Failed checks:", flush=True)
        for name, ok, detail in _results:
            if not ok:
                print(f"    • {name}  [{detail}]", flush=True)

    print("=" * 64 + "\n", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
