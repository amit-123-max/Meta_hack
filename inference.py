#!/usr/bin/env python3
"""
TrafficSignalEnv — OpenEnv Baseline Inference Script

Usage:
    python inference.py [--task easy|medium|hard|all] [--llm]

Environment variables:
    MODEL_PROVIDER   : "openai" | "hf" | "ollama"  (auto-detected)
    API_BASE_URL     : OpenAI-compatible API base URL
    MODEL_NAME       : Model name (also used as HF model if HF_MODEL_NAME unset)
    HF_TOKEN         : HuggingFace auth token
    HF_MODEL_NAME    : HF model name (falls back to MODEL_NAME)
    HF_API_URL / HF_ENDPOINT : Custom HF endpoint URL (optional)
    OPENROUTER_API_KEY: OpenRouter API key
    OLLAMA_MODEL     : If set, forces local Ollama endpoint

Output protocol:
    [START] task=<task_id>
    [STEP] step=<n> action=<action_json> reward=<r> done=<bool>
    [END] task=<task_id> score=<score>

Exit code 0 on success.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline.rule_based_agent import RuleBasedAgent
from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.medium_grader import MediumGrader
from llm_agent.llm_adapter import build_adapter, LLMAdapter
from tasks.task_easy import make_env as make_easy
from tasks.task_hard import make_env as make_hard
from tasks.task_medium import make_env as make_medium
from utils.replay import build_analytics, print_analytics


# ---------------------------------------------------------------------------
# LLM action helper (uses provider-agnostic adapter)
# ---------------------------------------------------------------------------

def _llm_choose_action(
    adapter: LLMAdapter,
    obs_meta: np.ndarray,
    n_intersections: int,
    task_id: str,
    step_feedback: Optional[Any] = None,
) -> List[int]:
    """Ask LLM to choose a phase for each intersection via the adapter."""
    import re

    meta_str = json.dumps(obs_meta.tolist(), separators=(",", ":"))
    fb_str   = ""
    if step_feedback is not None and hasattr(step_feedback, "to_compact_str"):
        risk = getattr(step_feedback, "risk_level", "low")
        if risk in ("medium", "high", "critical"):
            fb_str = f"\n{step_feedback.to_compact_str()}"

    prompt = (
        f"You control traffic lights for {n_intersections} intersection(s). "
        f"Task: {task_id}. "
        f"Metadata shape={obs_meta.shape}: {meta_str}. "
        "Each row: [q_N,q_S,q_E,q_W, phase, phase_timer, yellow_rem, "
        "emergency_type, emergency_lane, weather, spillback]. "
        "Choose phase for each intersection: 0=NS_GREEN, 1=EW_GREEN, 2=ALL_RED. "
        "If emergency_type>0, serve its lane. "
        f"Reply with exactly {n_intersections} comma-separated integers (0, 1, or 2)."
        + fb_str
    )

    try:
        text = adapter.complete(
            system="Output ONLY comma-separated integers 0-2. No words.",
            user=prompt,
            max_tokens=32,
            temperature=0.0,
        )
        if text is None:
            return [0] * n_intersections
        phases = [int(x.strip()) for x in re.findall(r"[0-2]", text)]
        phases = (phases + [0] * n_intersections)[:n_intersections]
        return phases
    except Exception as exc:
        print(f"[WARN] LLM call failed ({exc}); using fallback.", file=sys.stderr)
        return [0] * n_intersections


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    use_llm: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> float:
    """Run one episode and return the grader score."""
    if task_id == "easy":
        env    = make_easy(seed=seed)
        grader = EasyGrader()
    elif task_id == "medium":
        env    = make_medium(seed=seed)
        grader = MediumGrader()
    elif task_id == "hard":
        env    = make_hard(seed=seed)
        grader = HardGrader()
    else:
        raise ValueError(f"Unknown task_id: {task_id}")

    # Rule-based fallback agent
    agent = RuleBasedAgent(
        n_intersections=env.n_intersections,
        min_phase_steps=env.cfg.sim.phase_duration_min,
        max_phase_steps=env.cfg.sim.phase_duration_max,
    )

    # LLM adapter (provider-agnostic)
    adapter: Optional[LLMAdapter] = None
    if use_llm:
        adapter = build_adapter(verbose=verbose)
        if adapter is None:
            print("[WARN] No LLM adapter available — using rule-based agent.", flush=True)

    print(f"[START] task={task_id}", flush=True)

    obs = env.reset(seed=seed)
    agent.reset()
    done         = False
    step         = 0
    step_feedback = None

    while not done:
        if adapter is not None:
            phases = _llm_choose_action(
                adapter, obs.metadata,
                env.n_intersections, task_id,
                step_feedback=step_feedback,
            )
            action = phases
        else:
            ta     = agent.act(obs)
            action = ta.phase_indices

        obs, reward, done, info = env.step(action)
        step += 1

        # Capture step feedback for next iteration's LLM prompt
        step_feedback = info.get("step_feedback")

        action_json = json.dumps(action, separators=(",", ":"))
        print(
            f"[STEP] step={step} action={action_json} "
            f"reward={reward:.4f} done={done}",
            flush=True,
        )

    # Grade episode
    score = grader.grade(env.trajectory)

    # Analytics (uses real emergency data now)
    analytics = build_analytics(env.trajectory, task_id)

    # Episode feedback from env
    ep_feedback = info.get("episode_feedback")

    print(f"[END] task={task_id} score={score:.4f}", flush=True)

    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"  Task: {task_id.upper()}", flush=True)
        print(f"  Steps: {step}", flush=True)
        print(f"  Score: {score:.4f}", flush=True)
        print(f"  Episode Reward: {info['episode_reward']:.4f}", flush=True)
        if ep_feedback:
            print(f"  Lessons:", flush=True)
            for lesson in getattr(ep_feedback, "lessons", []):
                print(f"    • {lesson}", flush=True)
        print(f"\n  Analytics:", flush=True)
        print_analytics(analytics)
        print(f"{'='*60}\n", flush=True)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrafficSignalEnv baseline inference"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM agent (requires MODEL_PROVIDER / API_BASE_URL env var)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores: Dict[str, float] = {}

    start_time = time.time()

    for task_id in tasks:
        score = run_task(
            task_id=task_id,
            use_llm=args.llm,
            seed=args.seed,
            verbose=True,
        )
        scores[task_id] = score

    elapsed = time.time() - start_time

    print("\n" + "=" * 60, flush=True)
    print("  FINAL SCORES", flush=True)
    print("=" * 60, flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id:<10} {score:.4f}", flush=True)
    print(f"\n  Total runtime: {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
