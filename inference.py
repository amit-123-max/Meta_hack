#!/usr/bin/env python3
"""
TrafficSignalEnv — OpenEnv Baseline Inference Script

Usage:
    python inference.py [--task easy|medium|hard|all] [--llm]

Environment variables:
    API_BASE_URL  : OpenAI-compatible API base URL (for LLM mode)
    MODEL_NAME    : Model name to call (default: gpt-4o-mini)
    HF_TOKEN      : Hugging Face token (unused in inference but read for spec compliance)

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
# Ensure project root is on path (handles both direct + module execution)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline.rule_based_agent import RuleBasedAgent
from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.medium_grader import MediumGrader
from tasks.task_easy import make_env as make_easy
from tasks.task_hard import make_env as make_hard
from tasks.task_medium import make_env as make_medium
from utils.replay import build_analytics, print_analytics


# ---------------------------------------------------------------------------
# LLM client (optional, falls back to rule-based if API_BASE_URL not set)
# ---------------------------------------------------------------------------

def _get_llm_client() -> Optional[Any]:
    """Return an OpenAI client if API_BASE_URL is set, else None."""
    base_url = os.environ.get("API_BASE_URL", "")
    if not base_url:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(
            base_url=base_url,
            api_key=os.environ.get("HF_TOKEN"),
        )
    except ImportError:
        print("[WARN] openai package not found; falling back to rule-based agent.")
        return None


def _llm_choose_action(
    client: Any,
    model_name: str,
    obs_meta: np.ndarray,
    n_intersections: int,
    task_id: str,
) -> List[int]:
    """Ask LLM to choose a phase for each intersection.

    Falls back to random on error.
    """
    meta_str = json.dumps(obs_meta.tolist(), separators=(",", ":"))
    prompt = (
        f"You control traffic lights for {n_intersections} intersection(s). "
        f"Task: {task_id}. "
        f"Metadata (normalised, shape={obs_meta.shape}): {meta_str}. "
        "Each row: [q_N,q_S,q_E,q_W, phase, phase_timer, yellow_rem, "
        "emergency_type, emergency_lane, weather, spillback]. "
        "Choose phase for each intersection: 0=NS_GREEN, 1=EW_GREEN, 2=ALL_RED. "
        "If an emergency is present (emergency_type>0), prioritise serving its lane. "
        f"Reply with exactly {n_intersections} comma-separated integers (0, 1, or 2)."
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        phases = [int(x.strip()) for x in text.split(",")]
        phases = [max(0, min(2, p)) for p in phases]
        # Pad / trim
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
    # Build env
    if task_id == "easy":
        env     = make_easy(seed=seed)
        grader  = EasyGrader()
    elif task_id == "medium":
        env    = make_medium(seed=seed)
        grader = MediumGrader()
    elif task_id == "hard":
        env    = make_hard(seed=seed)
        grader = HardGrader()
    else:
        raise ValueError(f"Unknown task_id: {task_id}")

    # Agent
    agent = RuleBasedAgent(
        n_intersections=env.n_intersections,
        min_phase_steps=env.cfg.sim.phase_duration_min,
        max_phase_steps=env.cfg.sim.phase_duration_max,
    )
    llm_client  = _get_llm_client() if use_llm else None
    model_name  = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    print(f"[START] task={task_id}", flush=True)

    obs = env.reset(seed=seed)
    agent.reset()
    done = False
    step = 0

    while not done:
        # Choose action
        if llm_client is not None:
            phases = _llm_choose_action(
                llm_client, model_name, obs.metadata,
                env.n_intersections, task_id
            )
            action = phases
        else:
            ta     = agent.act(obs)
            action = ta.phase_indices

        obs, reward, done, info = env.step(action)
        step += 1

        action_json = json.dumps(action, separators=(",", ":"))
        print(
            f"[STEP] step={step} action={action_json} "
            f"reward={reward:.4f} done={done}",
            flush=True,
        )

    # Grade episode
    score = grader.grade(env.trajectory)

    # Analytics
    analytics = build_analytics(env.trajectory, task_id)

    print(f"[END] task={task_id} score={score:.4f}", flush=True)

    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"  Task: {task_id.upper()}", flush=True)
        print(f"  Steps: {step}", flush=True)
        print(f"  Score: {score:.4f}", flush=True)
        print(f"  Episode Reward: {info['episode_reward']:.4f}", flush=True)
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
        help="Use LLM agent (requires API_BASE_URL env var)",
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
