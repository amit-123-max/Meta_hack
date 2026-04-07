"""Trainer — multi-episode feedback loop for the LLM traffic agent.

Usage
-----
    from llm_agent.trainer import Trainer
    from tasks.task_easy import make_env
    from graders.easy_grader import EasyGrader

    trainer = Trainer(task_id="easy", n_episodes=5)
    results = trainer.train()

Or from CLI:
    python -m llm_agent.trainer --task easy --episodes 5

Learning loop (per episode)
---------------------------
1. env.reset()
2. For each step:
   a. agent.act(obs) → actions
   b. env.step(actions) → obs, reward, done, info
   c. agent.record_reward(reward, step)   ← feedback injected into memory
3. grader.grade(trajectory) → score
4. agent.end_episode(total_reward, score, n_steps) ← lessons extracted
5. Print rich progress log

After N episodes the agent's memory contains:
  - Top good / bad decisions (step-level)
  - Situational patterns keyed by tags
  - Episode summaries with lessons
  - Trend direction

These are all injected into subsequent episode prompts automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=False)  # Never override grader-injected env vars

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baseline.rule_based_agent import RuleBasedAgent
from graders.easy_grader   import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader   import HardGrader
from llm_agent.agent  import LLMAgent
from llm_agent.memory import AgentMemory
from tasks.task_easy   import make_env as make_easy
from tasks.task_medium import make_env as make_medium
from tasks.task_hard   import make_env as make_hard


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode:      int
    total_reward: float
    grader_score: float
    n_steps:      int
    elapsed_secs: float
    improvement:  Optional[float] = None   # vs. previous episode


@dataclass
class TrainingReport:
    task_id:        str
    n_episodes:     int
    results:        List[EpisodeResult] = field(default_factory=list)
    baseline_score: Optional[float]     = None

    @property
    def best_score(self) -> float:
        return max((r.grader_score for r in self.results), default=0.0)

    @property
    def final_score(self) -> float:
        return self.results[-1].grader_score if self.results else 0.0

    @property
    def improved(self) -> bool:
        if len(self.results) < 2:
            return False
        return self.results[-1].grader_score > self.results[0].grader_score


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_env_and_grader(task_id: str, seed: int):
    """Return (env, grader) for the given task."""
    if task_id == "easy":
        return make_easy(seed=seed), EasyGrader()
    if task_id == "medium":
        return make_medium(seed=seed), MediumGrader()
    if task_id == "hard":
        return make_hard(seed=seed), HardGrader()
    raise ValueError(f"Unknown task_id: {task_id!r}. Choose easy | medium | hard")


def _run_rule_based(task_id: str, seed: int) -> float:
    """Run one episode with the rule-based agent; return grader score."""
    env, grader = _make_env_and_grader(task_id, seed)
    agent = RuleBasedAgent(
        n_intersections=env.n_intersections,
        min_phase_steps=env.cfg.sim.phase_duration_min,
        max_phase_steps=env.cfg.sim.phase_duration_max,
    )
    obs = env.reset(seed=seed)
    agent.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, _, done, _ = env.step(action.phase_indices)
    return grader.grade(env.trajectory)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Runs the self-improving LLM agent training loop.

    Parameters
    ----------
    task_id     : "easy" | "medium" | "hard"
    n_episodes  : how many training episodes to run
    seed        : base random seed (increments per episode for variety)
    memory_path : optional JSON file to persist/load memory across runs
    verbose     : enable per-step logging from the LLM agent
    eval_baseline: if True, run rule-based baseline before training
    """

    def __init__(
        self,
        task_id:        str  = "easy",
        n_episodes:     int  = 5,
        seed:           int  = 42,
        memory_path:    Optional[str] = None,
        verbose:        bool = True,
        eval_baseline:  bool = True,
    ) -> None:
        self.task_id       = task_id
        self.n_episodes    = n_episodes
        self.base_seed     = seed
        self.memory_path   = memory_path
        self.verbose       = verbose
        self.eval_baseline = eval_baseline

        # Shared memory across all episodes
        self.memory = AgentMemory(persistence_path=memory_path)

        # Create a temporary env just to get n_intersections
        _env, _ = _make_env_and_grader(task_id, seed)
        self._n_intersections = _env.n_intersections

        # Single agent instance (memory persists across episodes)
        self.agent = LLMAgent(
            n_intersections=self._n_intersections,
            memory=self.memory,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Public: train
    # ------------------------------------------------------------------

    def train(self) -> TrainingReport:
        """Run n_episodes of the training loop. Returns TrainingReport."""
        report = TrainingReport(task_id=self.task_id, n_episodes=self.n_episodes)

        # Optional baseline measurement
        if self.eval_baseline:
            self._print_header("BASELINE (RuleBasedAgent)")
            baseline_score = _run_rule_based(self.task_id, self.base_seed)
            report.baseline_score = baseline_score
            print(f"  Baseline score: {baseline_score:.4f}")
            self._print_sep()

        # Training loop
        self._print_header(
            f"TRAINING — task={self.task_id.upper()}  episodes={self.n_episodes}"
        )

        for ep in range(1, self.n_episodes + 1):
            seed = self.base_seed + ep   # vary seed per episode
            result = self._run_episode(ep, seed)
            report.results.append(result)
            self._print_episode_result(result, report)

        self._print_final_report(report)
        return report

    # ------------------------------------------------------------------
    # Public: evaluate (no memory updates)
    # ------------------------------------------------------------------

    def evaluate(self, n_eval_episodes: int = 3, seed_offset: int = 100) -> Dict:
        """Run the LLM agent WITHOUT updating memory. Compare to baseline."""
        self._print_header("EVALUATION MODE (memory frozen)")

        llm_scores:  List[float] = []
        rule_scores: List[float] = []

        for ep in range(1, n_eval_episodes + 1):
            seed = self.base_seed + seed_offset + ep
            env, grader = _make_env_and_grader(self.task_id, seed)

            # LLM
            self.agent.reset(episode=ep)
            obs  = env.reset(seed=seed)
            done = False
            step = 0
            while not done:
                action = self.agent.act(obs, step=step)
                obs, reward, done, info = env.step(action)
                step += 1
            llm_score = grader.grade(env.trajectory)
            llm_scores.append(llm_score)

            # Rule-based on same seed
            rule_score = _run_rule_based(self.task_id, seed)
            rule_scores.append(rule_score)

            print(
                f"  Eval Ep {ep}: LLM={llm_score:.4f}  "
                f"Rule={rule_score:.4f}  "
                f"Δ={llm_score - rule_score:+.4f}"
            )

        avg_llm  = float(np.mean(llm_scores))
        avg_rule = float(np.mean(rule_scores))
        self._print_sep()
        print(f"  LLM  avg: {avg_llm:.4f}")
        print(f"  Rule avg: {avg_rule:.4f}")
        print(f"  Delta:    {avg_llm - avg_rule:+.4f}")
        self._print_sep()

        return {
            "llm_scores":  llm_scores,
            "rule_scores": rule_scores,
            "avg_llm":     avg_llm,
            "avg_rule":    avg_rule,
            "delta":       avg_llm - avg_rule,
        }

    # ------------------------------------------------------------------
    # Single episode runner
    # ------------------------------------------------------------------

    def _run_episode(self, episode: int, seed: int) -> EpisodeResult:
        env, grader = _make_env_and_grader(self.task_id, seed)
        self.agent.reset(episode=episode)

        obs  = env.reset(seed=seed)
        done = False
        step = 0
        total_reward = 0.0
        t0 = time.time()

        while not done:
            action = self.agent.act(obs, step=step)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Feed reward back to agent IMMEDIATELY
            self.agent.record_reward(reward, step=step)

            step += 1
            if self.verbose and step % 50 == 0:
                print(
                    f"    [Ep {episode} | Step {step}]  "
                    f"reward={reward:.4f}  cumulative={total_reward:.3f}",
                    flush=True,
                )

        # Grade
        grader_score = grader.grade(env.trajectory)

        # Update agent memory with lessons
        self.agent.end_episode(
            total_reward=total_reward,
            grader_score=grader_score,
            n_steps=step,
        )

        elapsed = time.time() - t0
        improvement = None
        if hasattr(self, "_prev_score"):
            improvement = grader_score - self._prev_score
        self._prev_score = grader_score

        return EpisodeResult(
            episode=episode,
            total_reward=total_reward,
            grader_score=grader_score,
            n_steps=step,
            elapsed_secs=elapsed,
            improvement=improvement,
        )

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _print_episode_result(
        self, result: EpisodeResult, report: TrainingReport
    ) -> None:
        imp_str = ""
        if result.improvement is not None:
            arrow = "↑" if result.improvement >= 0 else "↓"
            imp_str = f" ({arrow}{abs(result.improvement):.4f})"

        bar = self._score_bar(result.grader_score)
        print(
            f"\n  Episode {result.episode:>2}/{self.n_episodes}"
            f"  score={result.grader_score:.4f}{imp_str}"
            f"  reward={result.total_reward:>8.2f}"
            f"  steps={result.n_steps}"
            f"  {bar}"
            f"  [{result.elapsed_secs:.1f}s]",
            flush=True,
        )

        # Memory trend
        if len(self.memory.episode_summaries) > 1:
            print(f"  Trend: {self.memory.trend_string()}", flush=True)

        # Lessons from this episode
        if self.memory.episode_summaries:
            latest = self.memory.episode_summaries[-1]
            for lesson in latest.key_lessons:
                print(f"  💡 {lesson}", flush=True)

    def _print_final_report(self, report: TrainingReport) -> None:
        self._print_sep()
        print("  TRAINING COMPLETE", flush=True)
        self._print_sep()
        print(f"  Task:           {report.task_id.upper()}", flush=True)
        print(f"  Episodes:       {report.n_episodes}", flush=True)
        if report.baseline_score is not None:
            print(f"  Baseline score: {report.baseline_score:.4f}", flush=True)
        print(f"  Best score:     {report.best_score:.4f}", flush=True)
        print(f"  Final score:    {report.final_score:.4f}", flush=True)
        if report.baseline_score is not None:
            delta = report.final_score - report.baseline_score
            print(
                f"  vs Baseline:    {delta:+.4f} "
                f"({'BETTER' if delta >= 0 else 'WORSE'})",
                flush=True,
            )
        print(
            f"  Improved:       {'YES ✓' if report.improved else 'NO ✗'} "
            f"(ep1→final: "
            f"{report.results[0].grader_score:.4f} → {report.final_score:.4f})",
            flush=True,
        )
        self._print_sep()

        # Score table
        print("\n  Episode scores:", flush=True)
        print(f"  {'Ep':>4}  {'Score':>7}  {'Reward':>9}  {'Steps':>6}  Bar", flush=True)
        for r in report.results:
            bar = self._score_bar(r.grader_score)
            print(
                f"  {r.episode:>4}  {r.grader_score:>7.4f}  "
                f"{r.total_reward:>9.2f}  {r.n_steps:>6}  {bar}",
                flush=True,
            )
        self._print_sep()

    @staticmethod
    def _score_bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    @staticmethod
    def _print_header(title: str) -> None:
        print(f"\n{'='*60}", flush=True)
        print(f"  {title}", flush=True)
        print(f"{'='*60}", flush=True)

    @staticmethod
    def _print_sep() -> None:
        print(f"{'─'*60}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train or evaluate the LLM traffic signal agent."
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
        help="JSON file to persist agent memory (optional)",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only (no memory updates); requires --memory-path with existing data",
    )
    p.add_argument(
        "--eval-episodes", type=int, default=3,
        help="Number of episodes for evaluation mode (default: 3)",
    )
    p.add_argument(
        "--no-baseline", action="store_true",
        help="Skip rule-based baseline measurement",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step LLM logs",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    trainer = Trainer(
        task_id=args.task,
        n_episodes=args.episodes,
        seed=args.seed,
        memory_path=args.memory_path,
        verbose=not args.quiet,
        eval_baseline=not args.no_baseline,
    )

    if args.eval_only:
        trainer.evaluate(n_eval_episodes=args.eval_episodes)
    else:
        trainer.train()
        print("\n[Done] Run with --eval-only to compare LLM vs baseline.", flush=True)
