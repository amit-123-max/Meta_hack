#!/usr/bin/env python3
"""
run_llm_agent.py — Convenience entry point for LLM agent training & evaluation.

Usage examples
--------------
# Test LLM connectivity first (always do this!)
python run_llm_agent.py --test-llm

# Basic training on "easy" task, 3 episodes
python run_llm_agent.py --task easy --episodes 3

# Train with persistent memory (resumes from previous run)
python run_llm_agent.py --task easy --episodes 10 --memory-path memory_easy.json

# Evaluate after training (no memory updates)
python run_llm_agent.py --task easy --eval-only --memory-path memory_easy.json --eval-episodes 3

# Quiet mode (suppress per-step LLM logs)
python run_llm_agent.py --task medium --episodes 5 --quiet

Environment variables (set in .env or export)
----------------------------------------------
  OPENROUTER_API_KEY  — your OpenRouter API key  (preferred)
  HF_TOKEN            — HuggingFace token         (fallback)
  API_BASE_URL        — API endpoint  (default: https://openrouter.ai/api/v1)
  MODEL_NAME          — model ID      (default: openrouter/auto)
"""
from __future__ import annotations

import argparse
import os
import sys

# Make sure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=False)  # Never override grader-injected env vars

from llm_agent.agent import LLMAgent, llm_health_check
from llm_agent.llm_adapter import build_adapter


def _resolve_api_key() -> str:
    return (
        os.environ.get("API_KEY", "")
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_TOKEN", "")
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train or evaluate the LLM traffic signal agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", choices=["easy", "medium", "hard"], default="easy",
        help="Which task to run (default: easy)",
    )
    p.add_argument(
        "--episodes", type=int, default=5,
        help="Number of training episodes (default: 5)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    p.add_argument(
        "--memory-path", type=str, default=None,
        help="JSON file to persist agent memory across runs (optional)",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Evaluate without updating memory (requires --memory-path with existing data)",
    )
    p.add_argument(
        "--eval-episodes", type=int, default=3,
        help="Number of evaluation episodes (default: 3)",
    )
    p.add_argument(
        "--no-baseline", action="store_true",
        help="Skip rule-based baseline measurement",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step LLM logs (keeps episode summaries)",
    )
    p.add_argument(
        "--test-llm", action="store_true",
        help="Run LLM connectivity test only, then exit",
    )
    p.add_argument(
        "--skip-health-check", action="store_true",
        help="Skip the pre-training LLM health check (not recommended)",
    )
    p.add_argument(
        "--fallback-only", action="store_true",
        help="Skip ALL LLM calls and run purely with the rule-based fallback heuristic. "
             "Useful for offline testing and benchmarking without an LLM backend.",
    )
    return p.parse_args()


def run_health_check(verbose: bool = True) -> bool:
    """Run LLM health check. Returns True if LLM is reachable."""
    adapter = build_adapter(verbose=verbose)
    ok = llm_health_check(adapter=adapter, verbose=verbose)
    if ok:
        print("✔ LLM connected — proceeding.\n", flush=True)
    else:
        print(
            "❌ LLM failed — check your API key and API_BASE_URL in .env\n"
            "   Hint: run  python run_llm_agent.py --test-llm  for details.\n",
            flush=True,
        )
    return ok


def main() -> None:
    args = _parse_args()

    if args.test_llm:
        print("=" * 60)
        print("  LLM CONNECTIVITY TEST")
        print("=" * 60)
        print(f"  Provider     : {os.environ.get('MODEL_PROVIDER', 'auto')}")
        key = _resolve_api_key()
        print(f"  API key      : {'SET (' + key[:8] + '...)' if key else 'NOT SET'}")
        print("─" * 60)
        ok = run_health_check(verbose=True)
        sys.exit(0 if ok else 1)

    # ── Pre-training health check ───────────────────────────────────────
    # Skip health check if user explicitly opted out or wants fallback-only mode
    llm_confirmed_dead = args.fallback_only

    if not args.skip_health_check and not args.fallback_only:
        ok = run_health_check(verbose=not args.quiet)
        if not ok:
            print(
                "LLM is not working — check API key or model.\n"
                "  • Fix Ollama: run  ollama serve  then  ollama pull llama3\n"
                "  • Or use OpenRouter: set OPENROUTER_API_KEY in .env\n"
                "  • Or run with --fallback-only to skip LLM entirely (fast offline mode)",
                flush=True,
            )
            resp = input("Continue with fallback heuristic only? [y/N]: ").strip().lower()
            if resp != "y":
                sys.exit(1)
            llm_confirmed_dead = True  # skip all LLM calls — no point retrying

    if args.fallback_only:
        print("🔄 Running in FALLBACK-ONLY mode (no LLM calls).", flush=True)

    # ── Import trainer here (after health check) ────────────────────────
    from llm_agent.trainer import Trainer

    trainer = Trainer(
        task_id=args.task,
        n_episodes=args.episodes,
        seed=args.seed,
        memory_path=args.memory_path,
        verbose=not args.quiet,
        eval_baseline=not args.no_baseline,
    )

    # If LLM is confirmed dead (connection failed or fallback-only mode),
    # mark the agent's endpoint as dead immediately to skip all LLM retry loops.
    if llm_confirmed_dead:
        trainer.agent._llm_dead = True
        if not args.fallback_only:
            print("⚡ LLM marked as dead — all steps will use the rule-based fallback.",
                  flush=True)

    if args.eval_only:
        trainer.evaluate(n_eval_episodes=args.eval_episodes)
        return

    report = trainer.train()

    if not report.improved and report.n_episodes > 1:
        print(
            "\n[WARN] Agent score did not improve over training.\n"
            "       This can happen with very few episodes — try more.\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
