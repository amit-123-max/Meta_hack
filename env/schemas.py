"""Typed schemas for TrafficSignalEnv — Observation, Action, Reward, State, Feedback."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PhaseEnum(IntEnum):
    """Traffic light phase for a single intersection."""
    NS_GREEN = 0   # North-South green, East-West red
    EW_GREEN = 1   # East-West green, North-South red
    ALL_RED  = 2   # All red (safety / emergency transition)
    YELLOW   = 3   # Yellow (transition — handled internally)


class EmergencyType(IntEnum):
    """Emergency vehicle priority (higher value = higher priority)."""
    NONE      = 0
    POLICE    = 1
    FIRE      = 2
    AMBULANCE = 3


class WeatherCondition(IntEnum):
    """Visibility condition for camera noise augmentation."""
    CLEAR  = 0
    CLOUDY = 1
    RAIN   = 2
    FOG    = 3
    NIGHT  = 4


# ---------------------------------------------------------------------------
# Per-lane and per-intersection state
# ---------------------------------------------------------------------------

@dataclass
class LaneState:
    lane_id: int
    direction: str          # "N", "S", "E", "W"
    queue_length: int       # vehicles waiting
    throughput: int         # vehicles passed this step
    wait_time: float        # cumulative wait (seconds) this step
    is_green: bool
    emergency: EmergencyType = EmergencyType.NONE
    is_occluded: bool = False   # camera blocked


@dataclass
class IntersectionState:
    intersection_id: int
    lanes: List[LaneState]
    current_phase: PhaseEnum
    phase_timer: int        # steps since last phase change
    yellow_remaining: int   # steps of yellow transition remaining (0 = none)
    emergency_active: EmergencyType = EmergencyType.NONE
    emergency_lane: int = -1   # which lane has the emergency vehicle (-1 = none)
    weather: WeatherCondition = WeatherCondition.CLEAR
    total_throughput: int = 0
    total_wait: float = 0.0
    spillback_count: int = 0
    all_red_steps: int = 0     # total steps this intersection spent in ALL_RED


@dataclass
class TrafficState:
    """Full environment state returned by state()."""
    step: int
    intersections: List[IntersectionState]
    global_throughput: int
    global_avg_wait: float
    episode_emergency_delays: List[float]  # per-emergency response time (accurate)
    phase_switches: int                     # total across all intersections
    done: bool


# ---------------------------------------------------------------------------
# Observation (image frames + metadata)
# ---------------------------------------------------------------------------

@dataclass
class TrafficObservation:
    """What the agent sees at each step."""
    # Image stack: shape (frame_stack, H, W, 3) uint8
    frames: np.ndarray

    # Per-intersection metadata — shape (n_intersections, n_metadata_features)
    # Features per intersection (in order):
    #   [lane0_queue, lane1_queue, lane2_queue, lane3_queue,
    #    current_phase, phase_timer_norm, yellow_remaining,
    #    emergency_type, emergency_lane,
    #    weather, spillback_flag]
    metadata: np.ndarray

    # Convenience decoded fields (not fed to model directly)
    step: int = 0
    intersections: List[IntersectionState] = field(default_factory=list)

    # Dimensionality helpers
    @property
    def image_shape(self) -> Tuple[int, ...]:
        return self.frames.shape

    @property
    def metadata_shape(self) -> Tuple[int, ...]:
        return self.metadata.shape


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class TrafficAction:
    """Agent's action for one step.

    phase_indices:  list of PhaseEnum values, one per intersection.
                    If length < n_intersections, missing ones default to
                    holding current phase (safe no-op).
    extend_current: if True and the current phase is within valid timer bounds,
                    hold the current phase regardless of phase_indices.
    emergency_override: intersection index to force ALL_RED → then green for
                        the emergency lane; -1 means no override.
    """
    phase_indices: List[int]          # one per intersection
    extend_current: bool = False
    emergency_override: int = -1      # intersection idx; -1 = inactive

    @classmethod
    def from_flat_int(cls, action: int, n_intersections: int) -> "TrafficAction":
        """Decode a flat integer action into TrafficAction.

        Encoding (for n_intersections intersections, 3 phases each):
          action = sum over i of phase_i * (3 ** i)
        """
        phases = []
        remaining = action
        n_base_phases = len(PhaseEnum) - 1  # exclude YELLOW (internal)
        for _ in range(n_intersections):
            phases.append(remaining % n_base_phases)
            remaining //= n_base_phases
        return cls(phase_indices=phases)

    @classmethod
    def noop(cls, n_intersections: int) -> "TrafficAction":
        """No-op: keep all current phases."""
        return cls(phase_indices=[-1] * n_intersections)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class TrafficReward:
    """Component-wise reward breakdown for a single step."""
    throughput_bonus: float = 0.0
    queue_penalty: float = 0.0
    wait_penalty: float = 0.0
    switch_penalty: float = 0.0
    emergency_bonus: float = 0.0
    spillback_penalty: float = 0.0
    starvation_penalty: float = 0.0
    fairness_bonus: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.throughput_bonus
            + self.queue_penalty
            + self.wait_penalty
            + self.switch_penalty
            + self.emergency_bonus
            + self.spillback_penalty
            + self.starvation_penalty
            + self.fairness_bonus
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "throughput_bonus": self.throughput_bonus,
            "queue_penalty": self.queue_penalty,
            "wait_penalty": self.wait_penalty,
            "switch_penalty": self.switch_penalty,
            "emergency_bonus": self.emergency_bonus,
            "spillback_penalty": self.spillback_penalty,
            "starvation_penalty": self.starvation_penalty,
            "fairness_bonus": self.fairness_bonus,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Structured Feedback (Part B)
# ---------------------------------------------------------------------------

@dataclass
class StepFeedback:
    """Compact, deterministic per-step feedback from env to agent.

    Included in info["step_feedback"] on every env.step().
    All fields are deterministic given the same trajectory.
    """
    step: int
    risk_level: str                     # "low" | "medium" | "high" | "critical"
    dominant_queue: str                 # "NS" | "EW" | "balanced"
    emergency_active: bool
    emergency_type: str                 # e.g. "AMBULANCE", "NONE"
    spillback_active: bool
    starvation_detected: bool
    all_red_abused: bool                # True if ALL_RED chosen without justification
    last_action_sensible: bool          # did action serve dominant/emergency need?
    suggested_action: List[int]         # oracle heuristic recommendation
    reward_breakdown: Dict[str, float]  # component breakdown
    went_right: str                     # ≤80 char human-readable positive
    went_wrong: str                     # ≤80 char human-readable negative
    confidence: float                   # 1.0 = fully deterministic

    def to_compact_str(self) -> str:
        """Token-efficient string for LLM prompt injection (≤60 tokens)."""
        em = f"emerg={self.emergency_type}" if self.emergency_active else "emerg=none"
        issues = []
        if not self.last_action_sensible:
            issues.append("bad_action")
        if self.starvation_detected:
            issues.append("starving")
        if self.spillback_active:
            issues.append("spillback")
        if self.all_red_abused:
            issues.append("all_red_abuse")
        issue_str = ",".join(issues) if issues else "ok"
        rec = ",".join(str(a) for a in self.suggested_action)
        return (
            f"FEEDBACK[{self.step}]: risk={self.risk_level} {em} "
            f"dom={self.dominant_queue} issues={issue_str} rec=[{rec}]"
        )

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "risk_level": self.risk_level,
            "dominant_queue": self.dominant_queue,
            "emergency_active": self.emergency_active,
            "emergency_type": self.emergency_type,
            "spillback_active": self.spillback_active,
            "starvation_detected": self.starvation_detected,
            "all_red_abused": self.all_red_abused,
            "last_action_sensible": self.last_action_sensible,
            "suggested_action": self.suggested_action,
            "went_right": self.went_right,
            "went_wrong": self.went_wrong,
            "confidence": self.confidence,
            "reward_breakdown": self.reward_breakdown,
        }


@dataclass
class EpisodeFeedback:
    """Compact episode-level feedback produced at episode end.

    Included in info["episode_feedback"] when done=True.
    Available via env.build_episode_feedback() at any time.
    """
    n_steps: int
    avg_wait_per_lane: float
    total_throughput: int
    throughput_per_step: float
    emergency_events: List[Dict]         # each: {type, arrival, served, latency_steps}
    spillback_summary: Dict              # {mean_rate, max_rate, n_overflow_steps}
    violations: List[str]                # string descriptions of detected faults
    fairness_score: float                # Jain's index, episode-level [0,1]
    starvation_intersections: List[int]  # intersection IDs that were starved
    phase_churn_rate: float              # phase switches / (steps * n_intersections)
    all_red_rate: float                  # fraction of steps spent in ALL_RED
    best_step: int                       # step with highest reward
    worst_step: int                      # step with lowest reward
    lessons: List[str]                   # ≤5 concise actionable lessons
    score_breakdown: Dict[str, float]    # partial scores used by grader

    def to_dict(self) -> Dict:
        return {
            "n_steps": self.n_steps,
            "avg_wait_per_lane": self.avg_wait_per_lane,
            "total_throughput": self.total_throughput,
            "throughput_per_step": self.throughput_per_step,
            "emergency_events": self.emergency_events,
            "spillback_summary": self.spillback_summary,
            "violations": self.violations,
            "fairness_score": self.fairness_score,
            "starvation_intersections": self.starvation_intersections,
            "phase_churn_rate": self.phase_churn_rate,
            "all_red_rate": self.all_red_rate,
            "best_step": self.best_step,
            "worst_step": self.worst_step,
            "lessons": self.lessons,
            "score_breakdown": self.score_breakdown,
        }
