"""PromptBuilder — compact, token-efficient prompts for local LLMs (llama3/Ollama).

Metadata layout (per intersection, 11 features):
  idx  meaning
  ---  -------
  0    lane_N queue fraction  [0,1]
  1    lane_S queue fraction  [0,1]
  2    lane_E queue fraction  [0,1]
  3    lane_W queue fraction  [0,1]
  4    current phase norm     (0=NS_GREEN, ~0.33=EW_GREEN, ~0.67=ALL_RED)
  5    phase_timer norm       [0,1]
  6    yellow_remaining norm  [0,1]
  7    emergency_type norm    (0=none ... 1=ambulance)
  8    emergency_lane norm    [0,1] (decoded: int in -1..3)
  9    weather norm           (0=clear ... 1=night)
  10   spillback flag         {0,1}

Design principle: keep prompts SHORT for local models (llama3 via Ollama).
Target: system prompt < 60 tokens, user prompt < 400 tokens total.

v3 additions
------------
- build_user_prompt() accepts optional previous_action list for anti-repetition context.
- Explicit NS_total/EW_total comparison with deterministic decision rules.
- Anti-repetition enforcement: "Do NOT repeat same action blindly."
- Emergency override instruction included.
- Feedback block injects ≤3 lines when risk is non-trivial.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from llm_agent.memory import AgentMemory

# ------------------------------------------------------------------
# Decoding helpers
# ------------------------------------------------------------------
_PHASE_NAMES   = ["NS_GREEN", "EW_GREEN", "ALL_RED"]
_EMERG_NAMES   = ["NONE", "POLICE", "FIRE", "AMBULANCE"]
_WEATHER_NAMES = ["CLEAR", "CLOUDY", "RAIN", "FOG", "NIGHT"]
_LANE_NAMES    = ["N", "S", "E", "W"]

N_PHASES  = 3
N_EMERG   = 4
N_WEATHER = 5


def _decode_phase(norm: float) -> str:
    idx = min(N_PHASES - 1, max(0, round(norm * (N_PHASES - 1))))
    return _PHASE_NAMES[idx]


def _decode_emergency(norm: float) -> str:
    idx = min(N_EMERG - 1, max(0, round(norm * (N_EMERG - 1))))
    return _EMERG_NAMES[idx]


def _decode_weather(norm: float) -> str:
    idx = min(N_WEATHER - 1, max(0, round(norm * (N_WEATHER - 1))))
    return _WEATHER_NAMES[idx]


def _decode_emerg_lane(norm: float, n_lanes: int = 4) -> int:
    raw = round(norm * n_lanes) - 1
    return int(max(-1, min(n_lanes - 1, raw)))


def _queue_bar(q: float) -> str:
    """Single-char queue level: . L M H C"""
    if q < 0.15: return "."
    if q < 0.40: return "L"
    if q < 0.65: return "M"
    if q < 0.85: return "H"
    return "C"   # Critical


# ------------------------------------------------------------------
# Situation tagging
# ------------------------------------------------------------------

def extract_tags(meta_row: np.ndarray) -> List[str]:
    """Return situation tags for a single intersection metadata row."""
    tags: List[str] = []
    q     = meta_row[:4]
    emerg = float(meta_row[7])
    spill = float(meta_row[10])

    max_q = float(q.max())
    if max_q > 0.85:
        tags.append("critical_queue")
    elif max_q > 0.60:
        tags.append("high_queue")
    elif max_q > 0.25:
        tags.append("medium_queue")
    else:
        tags.append("low_queue")

    if emerg > 0.01:
        em_name = _decode_emergency(emerg).lower()
        tags.append("emergency")
        tags.append(f"emergency_{em_name}")

    if spill > 0.5:
        tags.append("spillback")

    ns_pressure = float(q[0] + q[1])
    ew_pressure = float(q[2] + q[3])
    if ns_pressure > ew_pressure * 1.5:
        tags.append("ns_dominant")
    elif ew_pressure > ns_pressure * 1.5:
        tags.append("ew_dominant")
    else:
        tags.append("balanced")

    return tags


# ------------------------------------------------------------------
# PromptBuilder
# ------------------------------------------------------------------

class PromptBuilder:
    """Builds compact, token-efficient prompts for local LLM agents.

    Design goals:
    - System prompt: < 60 tokens (llama3 likes short system prompts)
    - User prompt: < 350 tokens total
    - Output: ONLY the integer action list, nothing else
    """

    SYSTEM_PROMPT = (
        "You are a traffic signal controller. "
        "Output ONLY a comma-separated list of integers (0=NS_GREEN, 1=EW_GREEN). "
        "Follow queue comparison rules exactly. No explanation. No text."
    )

    def __init__(self, memory: AgentMemory, n_intersections: int) -> None:
        self.memory          = memory
        self.n_intersections = n_intersections

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def build_user_prompt(
        self,
        metadata:        np.ndarray,
        step:            int,
        episode:         int,
        all_tags:        Optional[List[List[str]]] = None,
        feedback:        Optional[Any] = None,      # StepFeedback | None
        previous_action: Optional[List[int]] = None,
    ) -> str:
        """Build a compact user prompt (target < 400 tokens).

        Sections (in order):
          1. STATE block — always included (with explicit NS/EW totals)
          2. CONTEXT block — previous action + anti-repetition warning
          3. FEEDBACK block — only when feedback is non-trivial (risk≥medium)
          4. LESSONS block — at most 2 lessons from memory
          5. ACTION instruction — always last (with deterministic decision rules)
        """
        parts: List[str] = []

        # 1. State (with explicit NS/EW totals for queue comparison)
        parts.append(self._format_state_compact(metadata))

        # 2. Previous action context + anti-repetition
        ctx_block = self._format_context_compact(previous_action)
        if ctx_block:
            parts.append(ctx_block)

        # 3. Feedback (compact, only when useful)
        if feedback is not None:
            fb_block = self._format_feedback_compact(feedback)
            if fb_block:
                parts.append(fb_block)

        # 4. Lessons from memory (only after first episode)
        insights = self.memory.get_insights()
        if insights["n_episodes"] > 0:
            lesson_block = self._format_lessons_compact(insights)
            if lesson_block:
                parts.append(lesson_block)

        # 5. Action instruction (deterministic rules)
        parts.append(self._format_action_compact(metadata, previous_action))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _format_state_compact(self, metadata: np.ndarray) -> str:
        """Compact state: one line per intersection with explicit queue totals."""
        lines = ["STATE:"]
        for i, row in enumerate(metadata):
            q          = row[:4]
            q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            ns         = q0 + q1
            ew         = q2 + q3
            phase_str  = _decode_phase(float(row[4]))
            yellow     = float(row[6]) > 0.05
            emerg_str  = _decode_emergency(float(row[7]))
            emerg_lane = _decode_emerg_lane(float(row[8]))
            spill      = float(row[10]) > 0.5

            flags = []
            if yellow:
                flags.append("YELLOW")
            if emerg_str != "NONE":
                elane = _LANE_NAMES[emerg_lane] if 0 <= emerg_lane < 4 else "?"
                flags.append(f"EMERG:{emerg_str}@{elane}")
            if spill:
                flags.append("SPILL")

            if ns > ew + 0.05:
                dom = "→NS wins"
            elif ew > ns + 0.05:
                dom = "→EW wins"
            else:
                dom = "→TIED(switch)"

            flag_str = " ".join(flags)
            # Explicit per-lane breakdown so LLM can compute NS_total/EW_total
            lines.append(
                f"  I{i}: N={q0:.2f} S={q1:.2f} E={q2:.2f} W={q3:.2f} "
                f"| NS_total={ns:.2f} EW_total={ew:.2f} {dom} "
                f"| phase={phase_str}"
                + (f" [{flag_str}]" if flags else "")
            )
        return "\n".join(lines)

    def _format_feedback_compact(self, feedback: Any) -> str:
        """Compact feedback block. Only included when risk is medium or above.

        Uses StepFeedback.to_compact_str() if available, else builds manually.
        Target: ≤ 3 lines, ≤ 60 tokens.
        """
        risk = getattr(feedback, "risk_level", "low")
        if risk == "low":
            return ""   # Don't bloat prompt when everything is fine

        if hasattr(feedback, "to_compact_str"):
            return feedback.to_compact_str()

        # Manual fallback for dict-style feedback
        lines = [f"FEEDBACK: risk={risk}"]
        went_wrong = getattr(feedback, "went_wrong", "")
        rec        = getattr(feedback, "suggested_action", [])
        if went_wrong:
            lines.append(f"  wrong: {went_wrong[:70]}")
        if rec:
            lines.append(f"  rec: [{','.join(str(a) for a in rec)}]")
        return "\n".join(lines)

    def _format_lessons_compact(self, insights: Dict[str, Any]) -> str:
        """At most 2 short lessons from memory."""
        lessons = insights.get("latest_lessons", [])
        # Also include recurring violations as a lesson hint
        violations = insights.get("recurring_violations", [])
        all_lessons = lessons + [f"VIOLATION: {v}" for v in violations[:1]]
        if not all_lessons:
            return ""
        top = [l[:80] for l in all_lessons[-2:]]
        return "LESSONS: " + " | ".join(top)

    def _format_context_compact(self, previous_action: Optional[List[int]]) -> str:
        """Compact previous-action block for anti-repetition context."""
        if previous_action is None:
            return ""
        labels = {0: "NS_GREEN", 1: "EW_GREEN", 2: "ALL_RED"}
        prev_str = ",".join(labels.get(a, str(a)) for a in previous_action)
        return (
            f"PREV_ACTION: {prev_str}. "
            "Do NOT repeat the same action blindly. Switch if unsure."
        )

    def _format_action_compact(
        self,
        metadata: np.ndarray,
        previous_action: Optional[List[int]] = None,
    ) -> str:
        """Structured action instruction with deterministic decision rules."""
        n = len(metadata)

        # Build per-intersection decision hints
        hints = []
        for i, row in enumerate(metadata):
            q     = row[:4]
            ns    = float(q[0]) + float(q[1])
            ew    = float(q[2]) + float(q[3])
            emerg = float(row[7])
            emerg_lane = _decode_emerg_lane(float(row[8]))
            prev  = previous_action[i] if previous_action and i < len(previous_action) else None

            if emerg > 0.01 and emerg_lane >= 0:
                phase = 0 if emerg_lane in [0, 1] else 1
                hints.append(f"I{i}→{phase}(EMERGENCY)")
            elif ns > ew + 0.05:
                hints.append(f"I{i}→0(NS>{ew:.2f})")
            elif ew > ns + 0.05:
                hints.append(f"I{i}→1(EW>{ns:.2f})")
            else:
                # Tied: switch from previous
                if prev == 0:
                    hints.append(f"I{i}→1(TIED,switch)")
                elif prev == 1:
                    hints.append(f"I{i}→0(TIED,switch)")
                else:
                    hints.append(f"I{i}→0(TIED)")

        hint_str = " | ".join(hints)
        return (
            f"RULES: IF NS_total>EW_total→0 | IF EW_total>NS_total→1 | IF equal→switch from prev.\n"
            f"HINT: {hint_str}\n"
            f"OUTPUT exactly {n} integer(s) 0 or 1, comma-separated. No words.\n"
            f"ANSWER:"
        )

    # ------------------------------------------------------------------
    # Fallback rule (used when LLM fails or circuit-breaker active)
    # ------------------------------------------------------------------

    def rule_fallback(self, metadata: np.ndarray) -> List[int]:
        """Best-effort heuristic fallback when LLM call fails."""
        actions = []
        for row in metadata:
            q          = row[:4]
            emerg      = float(row[7])
            emerg_lane = _decode_emerg_lane(float(row[8]))

            # Emergency override
            if emerg > 0.01 and emerg_lane >= 0:
                actions.append(0 if emerg_lane in [0, 1] else 1)
                continue

            # Yellow transition — hold current phase
            if float(row[6]) > 0.05:
                cur_phase = min(2, max(0, round(float(row[4]) * (N_PHASES - 1))))
                actions.append(int(cur_phase))
                continue

            # Pressure-based
            ns = float(q[0]) + float(q[1])
            ew = float(q[2]) + float(q[3])
            actions.append(0 if ns >= ew else 1)

        return actions
