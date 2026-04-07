"""AgentMemory — feedback-driven episodic memory for the LLM traffic agent.

Structure
---------
- good_decisions  : list of (situation_summary, action, reward, feedback_flags) for high-reward steps
- bad_decisions   : list of (situation_summary, action, reward, feedback_flags) for low-reward steps
- episode_summaries : per-episode roll-up (total_reward, key_lessons)
- situational_patterns : keyed by situation tag (e.g. "high_queue", "emergency")
  → best known action for that situation

The memory is intentionally lightweight (no embeddings, no vector DB).
It stays within a configurable max size by keeping only the most
informative decisions (sorted by |reward|).

New in v2
---------
- record_step() accepts optional StepFeedback for richer lesson extraction.
- record_episode() accepts optional EpisodeFeedback to pull structured lessons.
- StepRecord has two extra fields: risk_level, went_wrong.
- _extract_lessons() uses feedback flags in addition to raw reward.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
GOOD_REWARD_THRESHOLD = 0.65   # step reward above this → "good"
BAD_REWARD_THRESHOLD  = 0.40   # step reward below this → "bad"
MAX_GOOD_DECISIONS    = 20
MAX_BAD_DECISIONS     = 20
MAX_EPISODE_SUMMARIES = 30


@dataclass
class StepRecord:
    """A single (state, action, reward) experience tuple."""
    episode:    int
    step:       int
    situation:  str           # human-readable situation summary
    action:     List[int]     # phase choices (one per intersection)
    reward:     float
    tags:       List[str] = field(default_factory=list)
    # v2 additions — sourced from StepFeedback if available
    risk_level: str = "unknown"
    went_wrong: str = ""


@dataclass
class EpisodeSummary:
    """Roll-up of one completed episode."""
    episode:             int
    total_reward:        float
    n_steps:             int
    avg_reward:          float
    grader_score:        float
    key_lessons:         List[str] = field(default_factory=list)
    improvement_vs_prev: Optional[float] = None   # delta grader_score vs prev
    # v2 additions
    violations:          List[str] = field(default_factory=list)
    starvation_found:    bool = False
    emergency_neglect:   bool = False


class AgentMemory:
    """Persistent feedback-based memory.

    Usage
    -----
    >>> mem = AgentMemory()
    >>> mem.record_step(ep=1, step=5, situation="...", action=[0,1], reward=0.12)
    >>> mem.record_episode(ep=1, total_reward=-3.4, grader_score=0.55)
    >>> insights = mem.get_insights()
    """

    def __init__(
        self,
        good_thresh: float = GOOD_REWARD_THRESHOLD,
        bad_thresh: float  = BAD_REWARD_THRESHOLD,
        max_good: int      = MAX_GOOD_DECISIONS,
        max_bad: int       = MAX_BAD_DECISIONS,
        persistence_path: Optional[str] = None,
    ) -> None:
        self.good_thresh = good_thresh
        self.bad_thresh  = bad_thresh
        self.max_good    = max_good
        self.max_bad     = max_bad
        self.persistence_path = persistence_path

        self.good_decisions:       List[StepRecord]     = []
        self.bad_decisions:        List[StepRecord]     = []
        self.episode_summaries:    List[EpisodeSummary] = []
        self.situational_patterns: Dict[str, Dict]      = {}

        self._current_trajectory: List[StepRecord] = []

        if persistence_path and os.path.exists(persistence_path):
            self._load(persistence_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        episode:   int,
        step:      int,
        situation: str,
        action:    List[int],
        reward:    float,
        tags:      Optional[List[str]] = None,
        feedback:  Optional[Any] = None,   # StepFeedback | None
    ) -> None:
        """Record a single environment step.

        Args:
            feedback: optional StepFeedback dataclass from info["step_feedback"].
        """
        # Extract v2 fields from feedback if available
        risk_level = "unknown"
        went_wrong = ""
        if feedback is not None:
            risk_level = getattr(feedback, "risk_level", "unknown")
            went_wrong = getattr(feedback, "went_wrong", "")
            # Merge feedback-derived tags
            extra_tags: List[str] = []
            if getattr(feedback, "emergency_active", False):
                extra_tags.append("emergency")
            if getattr(feedback, "spillback_active", False):
                extra_tags.append("spillback")
            if getattr(feedback, "starvation_detected", False):
                extra_tags.append("starvation")
            if getattr(feedback, "all_red_abused", False):
                extra_tags.append("all_red_abuse")
            tags = list(set((tags or []) + extra_tags))

        record = StepRecord(
            episode=episode,
            step=step,
            situation=situation,
            action=action,
            reward=reward,
            tags=tags or [],
            risk_level=risk_level,
            went_wrong=went_wrong,
        )
        self._current_trajectory.append(record)

        if reward >= self.good_thresh:
            self.good_decisions.append(record)
            self.good_decisions.sort(key=lambda r: r.reward, reverse=True)
            self.good_decisions = self.good_decisions[: self.max_good]
        elif reward <= self.bad_thresh:
            self.bad_decisions.append(record)
            self.bad_decisions.sort(key=lambda r: r.reward)
            self.bad_decisions = self.bad_decisions[: self.max_bad]

        for tag in (tags or []):
            existing = self.situational_patterns.get(tag)
            if existing is None or reward > existing["best_reward"]:
                self.situational_patterns[tag] = {
                    "best_action":  action,
                    "best_reward":  reward,
                    "situation":    situation,
                    "episode":      episode,
                    "step":         step,
                }

    def record_episode(
        self,
        episode:          int,
        total_reward:     float,
        grader_score:     float,
        n_steps:          int,
        episode_feedback: Optional[Any] = None,   # EpisodeFeedback | None
    ) -> None:
        """Summarise a completed episode and extract lessons."""
        avg_reward = total_reward / max(n_steps, 1)

        improvement = None
        if self.episode_summaries:
            prev = self.episode_summaries[-1]
            improvement = grader_score - prev.grader_score

        # Merge feedback-based lessons
        fb_lessons: List[str] = []
        violations:  List[str] = []
        starvation_found  = False
        emergency_neglect = False

        if episode_feedback is not None:
            fb_lessons   = list(getattr(episode_feedback, "lessons", []))
            violations   = list(getattr(episode_feedback, "violations", []))
            starvation_found  = bool(getattr(episode_feedback, "starvation_intersections", []))
            em_events = getattr(episode_feedback, "emergency_events", [])
            emergency_neglect = any(not ev.get("served", True) for ev in em_events)

        traj_lessons = self._extract_lessons(total_reward, grader_score, episode_feedback)
        # Combine: structured feedback lessons first, then trajectory-derived
        all_lessons = (fb_lessons + traj_lessons)[:10]  # cap at 10

        summary = EpisodeSummary(
            episode=episode,
            total_reward=total_reward,
            n_steps=n_steps,
            avg_reward=avg_reward,
            grader_score=grader_score,
            key_lessons=all_lessons,
            improvement_vs_prev=improvement,
            violations=violations,
            starvation_found=starvation_found,
            emergency_neglect=emergency_neglect,
        )
        self.episode_summaries.append(summary)
        if len(self.episode_summaries) > MAX_EPISODE_SUMMARIES:
            self.episode_summaries = self.episode_summaries[-MAX_EPISODE_SUMMARIES:]

        self._current_trajectory = []

        if self.persistence_path:
            self._save(self.persistence_path)

    # ------------------------------------------------------------------
    # Insight extraction (used by PromptBuilder)
    # ------------------------------------------------------------------

    def get_insights(self) -> Dict[str, Any]:
        return {
            "n_episodes":            len(self.episode_summaries),
            "best_episode":          self._best_episode(),
            "worst_episode":         self._worst_episode(),
            "trend":                 self._compute_trend(),
            "top_good_decisions":    self._format_decisions(self.good_decisions[:5]),
            "top_bad_decisions":     self._format_decisions(self.bad_decisions[:5]),
            "situational_patterns":  self._format_patterns(),
            "latest_lessons":        self._latest_lessons(n=3),
            "recurring_violations":  self._recurring_violations(n=2),
        }

    def get_situational_advice(self, tags: List[str]) -> str:
        advice_lines = []
        for tag in tags:
            pattern = self.situational_patterns.get(tag)
            if pattern:
                advice_lines.append(
                    f"  [{tag}] Best known action: {pattern['best_action']} "
                    f"(reward={pattern['best_reward']:.3f})"
                )
        return "\n".join(advice_lines) if advice_lines else ""

    def trend_string(self) -> str:
        if len(self.episode_summaries) < 2:
            return "No trend yet (fewer than 2 episodes)."
        recent = self.episode_summaries[-5:]
        scores = [e.grader_score for e in recent]
        direction = "IMPROVING ↑" if scores[-1] > scores[0] else "DECLINING ↓"
        return f"Recent scores: {[f'{s:.3f}' for s in scores]}  → {direction}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_lessons(
        self,
        total_reward: float,
        score: float,
        episode_feedback: Optional[Any] = None,
    ) -> List[str]:
        """Derive text lessons from this episode's trajectory and feedback."""
        lessons: List[str] = []
        traj = self._current_trajectory

        # --- feedback-driven lessons (structured, more precise) ---
        if episode_feedback is not None:
            all_red_rate = getattr(episode_feedback, "all_red_rate", 0.0)
            churn_rate   = getattr(episode_feedback, "phase_churn_rate", 0.0)
            em_events    = getattr(episode_feedback, "emergency_events", [])
            starved      = getattr(episode_feedback, "starvation_intersections", [])

            if all_red_rate > 0.30:
                lessons.append(
                    f"ALL_RED used {all_red_rate:.0%} of steps — "
                    "avoid unless yellow/emergency is active."
                )
            if churn_rate > 0.40:
                lessons.append(
                    f"Phase switched too often ({churn_rate:.0%}). "
                    "Hold green for at least 5 steps."
                )
            unserved = [e for e in em_events if not e.get("served")]
            if unserved:
                lessons.append(
                    f"{len(unserved)} emergency(ies) left unserved. "
                    "Prioritise emergency lane immediately."
                )
            if starved:
                lessons.append(
                    f"Intersection(s) {starved} nearly starved. "
                    "Distribute green time fairly."
                )

        # --- trajectory-driven lessons (reward-based) ---
        if not traj:
            return lessons

        bad_streak = max_bad_streak = 0
        for rec in traj:
            if rec.reward < BAD_REWARD_THRESHOLD:
                bad_streak += 1
                max_bad_streak = max(max_bad_streak, bad_streak)
            else:
                bad_streak = 0
        if max_bad_streak > 5:
            lessons.append(
                f"Low-reward streak ({max_bad_streak} steps below {BAD_REWARD_THRESHOLD}). "
                "Serve the largest queue direction."
            )

        # ALL_RED abuse from step records
        all_red_abuse_steps = sum(1 for r in traj if "all_red_abuse" in r.tags)
        if all_red_abuse_steps > max(len(traj) * 0.10, 2):
            lessons.append(
                f"ALL_RED chosen without justification {all_red_abuse_steps}× — "
                "only use during yellow transitions or emergencies."
            )

        # Emergency handling from tagged steps
        emerg_recs = [r for r in traj if "emergency" in r.tags]
        if emerg_recs:
            avg_emerg_reward = sum(r.reward for r in emerg_recs) / len(emerg_recs)
            if avg_emerg_reward > BAD_REWARD_THRESHOLD:
                lessons.append("Emergency handling effective — keep prioritising emergency lanes.")
            else:
                lessons.append(
                    "Emergency handling poor — serve emergency lane "
                    "immediately when emergency_type > 0."
                )

        # Oscillation
        switches = sum(
            1 for i in range(1, len(traj)) if traj[i].action != traj[i - 1].action
        )
        switch_rate = switches / max(len(traj), 1)
        if switch_rate > 0.3:
            lessons.append(
                f"High switch rate ({switch_rate:.2f}). Hold phases longer."
            )
        elif switch_rate < 0.05:
            lessons.append(
                "Very low switch rate — phase held too long, starvation risk."
            )

        if score < 0.40:
            lessons.append(
                "Low score. Focus on throughput: serve the direction with "
                "the largest combined queue."
            )
        elif score > 0.65:
            lessons.append("High score — strategy working, maintain it.")

        return lessons

    def _recurring_violations(self, n: int = 2) -> List[str]:
        """Most common violations across recent episodes."""
        from collections import Counter
        recent = self.episode_summaries[-5:]
        all_v: List[str] = []
        for ep in recent:
            all_v.extend(ep.violations)
        counter = Counter(all_v)
        return [v for v, _ in counter.most_common(n)]

    def _best_episode(self) -> Optional[Dict]:
        if not self.episode_summaries:
            return None
        best = max(self.episode_summaries, key=lambda e: e.grader_score)
        return {"episode": best.episode, "grader_score": best.grader_score}

    def _worst_episode(self) -> Optional[Dict]:
        if not self.episode_summaries:
            return None
        worst = min(self.episode_summaries, key=lambda e: e.grader_score)
        return {"episode": worst.episode, "grader_score": worst.grader_score}

    def _compute_trend(self) -> str:
        return self.trend_string()

    def _format_decisions(self, records: List[StepRecord]) -> List[str]:
        out = []
        for r in records:
            parts = [f"Ep{r.episode}/Step{r.step}: action={r.action}, reward={r.reward:.4f}"]
            if r.went_wrong:
                parts.append(f"wrong: {r.went_wrong[:60]}")
            parts.append(r.situation[:80])
            out.append(" | ".join(parts))
        return out

    def _format_patterns(self) -> List[str]:
        lines = []
        for tag, info in self.situational_patterns.items():
            lines.append(
                f"  [{tag}] best_action={info['best_action']} "
                f"(reward={info['best_reward']:.3f})"
            )
        return lines

    def _latest_lessons(self, n: int = 3) -> List[str]:
        recent = self.episode_summaries[-n:]
        lessons: List[str] = []
        for ep in recent:
            lessons.extend(ep.key_lessons)
        return lessons[-10:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        try:
            data = {
                "good_decisions":       [asdict(r) for r in self.good_decisions],
                "bad_decisions":        [asdict(r) for r in self.bad_decisions],
                "episode_summaries":    [asdict(e) for e in self.episode_summaries],
                "situational_patterns": self.situational_patterns,
            }
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            print(f"[Memory] WARNING: could not save memory: {exc}")

    def _load(self, path: str) -> None:
        try:
            with open(path) as fh:
                data = json.load(fh)
            # Tolerate old records missing new fields (backward compat)
            good_raw = data.get("good_decisions", [])
            bad_raw  = data.get("bad_decisions",  [])
            self.good_decisions = [self._load_step_record(r) for r in good_raw]
            self.bad_decisions  = [self._load_step_record(r) for r in bad_raw]
            ep_raw = data.get("episode_summaries", [])
            self.episode_summaries = [self._load_ep_summary(e) for e in ep_raw]
            self.situational_patterns = data.get("situational_patterns", {})
            print(f"[Memory] Loaded {len(self.episode_summaries)} episode(s) from {path}")
        except Exception as exc:
            print(f"[Memory] WARNING: could not load memory: {exc}")

    @staticmethod
    def _load_step_record(d: Dict) -> StepRecord:
        """Load StepRecord, filling in missing v2 fields with defaults."""
        return StepRecord(
            episode=d.get("episode", 0),
            step=d.get("step", 0),
            situation=d.get("situation", ""),
            action=d.get("action", []),
            reward=d.get("reward", 0.0),
            tags=d.get("tags", []),
            risk_level=d.get("risk_level", "unknown"),
            went_wrong=d.get("went_wrong", ""),
        )

    @staticmethod
    def _load_ep_summary(d: Dict) -> EpisodeSummary:
        """Load EpisodeSummary, filling in missing v2 fields with defaults."""
        return EpisodeSummary(
            episode=d.get("episode", 0),
            total_reward=d.get("total_reward", 0.0),
            n_steps=d.get("n_steps", 0),
            avg_reward=d.get("avg_reward", 0.0),
            grader_score=d.get("grader_score", 0.0),
            key_lessons=d.get("key_lessons", []),
            improvement_vs_prev=d.get("improvement_vs_prev"),
            violations=d.get("violations", []),
            starvation_found=d.get("starvation_found", False),
            emergency_neglect=d.get("emergency_neglect", False),
        )
