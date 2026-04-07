"""Main TrafficSignalEnv — OpenEnv-compliant environment class."""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.env_config import EnvConfig
from env.observation import ObservationBuilder
from env.reward import compute_reward
from env.schemas import (
    EmergencyType,
    EpisodeFeedback,
    PhaseEnum,
    StepFeedback,
    TrafficAction,
    TrafficObservation,
    TrafficReward,
    TrafficState,
)
from env.simulator import TrafficSimulator

# Number of discrete phases the agent can choose per intersection
N_PHASES = 3  # NS_GREEN, EW_GREEN, ALL_RED

# Phase → lane group name
_PHASE_GROUP = {0: "NS", 1: "EW", 2: "ALL_RED"}
_EMERG_NAMES = {0: "NONE", 1: "POLICE", 2: "FIRE", 3: "AMBULANCE"}
_RISK_THRESHOLDS = {"critical": 0.85, "high": 0.60, "medium": 0.35}


def _safe_mean(vals: List[float], default: float = 0.0) -> float:
    return statistics.mean(vals) if vals else default


class TrafficSignalEnv:
    """OpenEnv-compliant adaptive traffic signal control environment.

    Observation space
    -----------------
    TrafficObservation:
      • frames   : uint8 array (frame_stack, H, W, 3)
      • metadata : float32 array (n_intersections, 11)

    Action space
    ------------
    TrafficAction — or any of these raw formats (all are accepted):
      • List[int]  : one phase index per intersection (0=NS_GREEN, 1=EW_GREEN, 2=ALL_RED)
      • int        : flat integer decoded via TrafficAction.from_flat_int()
      • TrafficAction: passed through directly

    Invalid actions are silently converted to no-ops.

    Reward
    ------
    Dense scalar (float) returned by step(); component breakdown in info["reward_breakdown"].

    Feedback (NEW)
    ------
    info["step_feedback"]   : StepFeedback dataclass (every step)
    info["episode_feedback"]: EpisodeFeedback dataclass (only when done=True)

    Episode
    -------
    An episode ends after SimConfig.max_steps steps (done=True).
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._sim = TrafficSimulator(cfg)
        self._obs_builder = ObservationBuilder(cfg)
        self._prev_state: Optional[TrafficState] = None
        self._prev_action: Optional[List[int]] = None
        self._trajectory: List[Dict] = []   # for graders / replay
        self._episode_reward: float = 0.0
        self._step_rewards: List[float] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> TrafficObservation:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self.cfg.sim.seed = seed
        self._sim.reset()
        self._obs_builder.reset()
        self._prev_state = None
        self._prev_action = None
        self._trajectory = []
        self._episode_reward = 0.0
        self._step_rewards = []

        state = self._sim.get_state()
        obs = self._obs_builder.build(state)
        self._prev_state = state
        return obs

    def step(
        self, action: Any
    ) -> Tuple[TrafficObservation, float, bool, Dict]:
        """Advance one step.

        Args:
            action: TrafficAction | List[int] | int

        Returns:
            (observation, reward, done, info)
            info always contains "step_feedback" (StepFeedback).
            info contains "episode_feedback" only when done=True.
        """
        parsed_action = self._parse_action(action)

        # Apply emergency override if requested
        phase_indices = self._resolve_emergency_override(parsed_action)

        # Step simulator
        step_stats = self._sim.step(phase_indices)

        # Get new state
        state = self._sim.get_state()

        # Compute reward
        rew_obj: TrafficReward = compute_reward(
            state=state,
            prev_state=self._prev_state,
            step_stats=step_stats,
            max_queue=self.cfg.sim.max_queue_per_lane,
        )
        reward = float(rew_obj.total)
        self._episode_reward += reward
        self._step_rewards.append(reward)

        # Build observation
        obs = self._obs_builder.build(state)

        done = state.done

        # Compute structured step feedback (deterministic)
        step_feedback = self._compute_step_feedback(
            state=state,
            action_taken=phase_indices,
            rew_obj=rew_obj,
            step=state.step,
        )

        info: Dict[str, Any] = {
            "step": state.step,
            "reward_breakdown": rew_obj.to_dict(),
            "global_throughput": state.global_throughput,
            "global_avg_wait": state.global_avg_wait,
            "phase_switches": state.phase_switches,
            "emergency_delays": list(state.episode_emergency_delays),
            "episode_reward": self._episode_reward,
            "step_feedback": step_feedback,
        }

        # Build episode-level feedback when episode ends
        if done:
            ep_feedback = self.build_episode_feedback()
            info["episode_feedback"] = ep_feedback

        # Record trajectory snapshot (extended with emergency events)
        snap = self._snapshot(state)
        self._trajectory.append({
            "step": state.step,
            "action": parsed_action,
            "reward": reward,
            "reward_breakdown": rew_obj.to_dict(),
            "state_snapshot": snap,
            "step_feedback": step_feedback.to_dict(),
        })

        self._prev_state = state
        self._prev_action = list(phase_indices)
        return obs, reward, done, info

    def state(self) -> TrafficState:
        """Return full current state (OpenEnv state() requirement)."""
        return self._sim.get_state()

    def render(self) -> np.ndarray:
        """Return current frame as numpy array (H, W, 3) uint8."""
        state = self._sim.get_state()
        obs = self._obs_builder.build(state)
        return obs.frames[-1]  # most recent frame

    # ------------------------------------------------------------------
    # Feedback computation
    # ------------------------------------------------------------------

    def _compute_step_feedback(
        self,
        state: TrafficState,
        action_taken: List[int],
        rew_obj: TrafficReward,
        step: int,
    ) -> StepFeedback:
        """Compute deterministic per-step feedback object."""
        n = self.cfg.n_intersections

        # --- Aggregate signals across all intersections ---
        max_queue_frac = 0.0
        global_ns_pressure = 0.0
        global_ew_pressure = 0.0
        emergency_active = False
        emergency_type_str = "NONE"
        spillback_active = False
        starvation_detected = False
        all_red_abused = False

        for inter in state.intersections:
            queues = [l.queue_length for l in inter.lanes]
            if queues:
                max_q = max(queues) / max(self.cfg.sim.max_queue_per_lane, 1)
                max_queue_frac = max(max_queue_frac, max_q)
            # NS = lanes 0,1; EW = lanes 2,3
            if len(inter.lanes) >= 4:
                global_ns_pressure += inter.lanes[0].queue_length + inter.lanes[1].queue_length
                global_ew_pressure += inter.lanes[2].queue_length + inter.lanes[3].queue_length

            em = inter.emergency_active
            if em != EmergencyType.NONE:
                emergency_active = True
                if em.value > EmergencyType[emergency_type_str].value if emergency_type_str != "NONE" else True:
                    emergency_type_str = em.name

            if inter.spillback_count > 0:
                spillback_active = True

        # Starvation: any lane starved > 8 steps in current state
        for inter in state.intersections:
            for lane in inter.lanes:
                # We can't read internal starvation_timer from TrafficState easily
                # so use queue proxy: high queue + not green = starvation candidate
                if lane.queue_length > int(self.cfg.sim.max_queue_per_lane * 0.6) and not lane.is_green:
                    starvation_detected = True

        # Risk level
        if emergency_active or max_queue_frac > _RISK_THRESHOLDS["critical"]:
            risk_level = "critical"
        elif spillback_active or max_queue_frac > _RISK_THRESHOLDS["high"]:
            risk_level = "high"
        elif max_queue_frac > _RISK_THRESHOLDS["medium"]:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Dominant queue direction
        if global_ns_pressure > global_ew_pressure * 1.3:
            dominant_queue = "NS"
        elif global_ew_pressure > global_ns_pressure * 1.3:
            dominant_queue = "EW"
        else:
            dominant_queue = "balanced"

        # Compute oracle suggested action
        suggested_action = self._oracle_action(state)

        # Was last action sensible?
        last_action_sensible = True
        for i, (taken, suggested) in enumerate(zip(action_taken[:n], suggested_action[:n])):
            if emergency_active:
                # During emergency, must NOT be ALL_RED without justification
                inter = state.intersections[i] if i < len(state.intersections) else None
                if inter and inter.emergency_active != EmergencyType.NONE:
                    em_lane = inter.emergency_lane
                    # Phase must serve the emergency lane
                    expected = 0 if em_lane in [0, 1] else 1
                    if taken == 2:  # ALL_RED during emergency
                        last_action_sensible = False
                        break
            if taken == 2:  # ALL_RED
                all_red_abused = True
                last_action_sensible = False

        # Generate went_right / went_wrong messages
        went_right = ""
        went_wrong = ""
        if rew_obj.throughput_bonus > 0.1:
            went_right = f"Good throughput (+{rew_obj.throughput_bonus:.2f})"
        elif rew_obj.emergency_bonus > 0.0:
            went_right = f"Emergency served (+{rew_obj.emergency_bonus:.2f})"
        else:
            went_right = "Step completed without crash."

        if all_red_abused:
            went_wrong = "ALL_RED chosen without emergency/yellow justification."
        elif starvation_detected:
            went_wrong = "High-queue lane left red — starvation risk."
        elif rew_obj.spillback_penalty < -0.05:
            went_wrong = f"Spillback penalty ({rew_obj.spillback_penalty:.2f})."
        elif rew_obj.queue_penalty < -0.1:
            went_wrong = f"High queue pressure ({rew_obj.queue_penalty:.2f})."
        else:
            went_wrong = ""

        return StepFeedback(
            step=step,
            risk_level=risk_level,
            dominant_queue=dominant_queue,
            emergency_active=emergency_active,
            emergency_type=emergency_type_str,
            spillback_active=spillback_active,
            starvation_detected=starvation_detected,
            all_red_abused=all_red_abused,
            last_action_sensible=last_action_sensible,
            suggested_action=suggested_action,
            reward_breakdown=rew_obj.to_dict(),
            went_right=went_right[:80],
            went_wrong=went_wrong[:80],
            confidence=1.0,  # deterministic
        )

    def _oracle_action(self, state: TrafficState) -> List[int]:
        """Heuristic oracle: serve emergency lane first, then dominant queue."""
        actions = []
        for inter in state.intersections:
            # Emergency priority
            em = inter.emergency_active
            if em != EmergencyType.NONE and inter.emergency_lane >= 0:
                actions.append(0 if inter.emergency_lane in [0, 1] else 1)
                continue
            # Yellow — hold current
            if inter.yellow_remaining > 0:
                actions.append(int(inter.current_phase.value) % 2)
                continue
            # Queue pressure
            if len(inter.lanes) >= 4:
                ns = inter.lanes[0].queue_length + inter.lanes[1].queue_length
                ew = inter.lanes[2].queue_length + inter.lanes[3].queue_length
                actions.append(0 if ns >= ew else 1)
            else:
                actions.append(0)
        return actions

    def build_episode_feedback(self) -> EpisodeFeedback:
        """Build episode-level feedback from the completed trajectory."""
        traj = self._trajectory
        n_steps = len(traj)
        if n_steps == 0:
            return EpisodeFeedback(
                n_steps=0, avg_wait_per_lane=0.0, total_throughput=0,
                throughput_per_step=0.0, emergency_events=[],
                spillback_summary={}, violations=[], fairness_score=1.0,
                starvation_intersections=[], phase_churn_rate=0.0,
                all_red_rate=0.0, best_step=0, worst_step=0,
                lessons=[], score_breakdown={},
            )

        # Aggregate metrics
        rewards = [s.get("reward", 0.0) for s in traj]
        throughputs = [int(s.get("state_snapshot", {}).get("global_throughput", 0)) for s in traj]
        avg_waits = [float(s.get("state_snapshot", {}).get("global_avg_wait", 0.0)) for s in traj]

        total_throughput = sum(throughputs)
        avg_wait = _safe_mean(avg_waits)
        best_step = int(np.argmax(rewards)) + 1
        worst_step = int(np.argmin(rewards)) + 1

        # Emergency events from simulator
        emergency_events = self._sim.get_served_emergency_events()

        # Spillback summary
        spill_steps = 0
        spill_rates: List[float] = []
        n_inters = max(self.cfg.n_intersections, 1)
        for step_data in traj:
            snap = step_data.get("state_snapshot", {})
            inter_list = snap.get("intersections", [])
            if inter_list:
                step_spills = sum(1 for i in inter_list if i.get("spillback", 0) > 0)
                rate = step_spills / n_inters
                spill_rates.append(rate)
                if step_spills > 0:
                    spill_steps += 1
        spillback_summary = {
            "mean_rate": round(_safe_mean(spill_rates), 4),
            "max_rate": round(max(spill_rates) if spill_rates else 0.0, 4),
            "n_overflow_steps": spill_steps,
        }

        # Phase churn
        final_switches = int(traj[-1].get("state_snapshot", {}).get("phase_switches", 0)) if traj else 0
        phase_churn_rate = final_switches / max(n_steps * n_inters, 1)

        # ALL_RED rate
        all_red_steps_total = 0
        for snap_data in traj:
            snap = snap_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                if inter.get("phase", -1) == PhaseEnum.ALL_RED.value:
                    all_red_steps_total += 1
        all_red_rate = all_red_steps_total / max(n_steps * n_inters, 1)

        # Episode-level Jain's fairness
        inter_totals: Dict[int, float] = {}
        for step_data in traj:
            snap = step_data.get("state_snapshot", {})
            for inter in snap.get("intersections", []):
                iid = inter.get("id", 0)
                itp = float(inter.get("throughput", 0.0))
                inter_totals[iid] = inter_totals.get(iid, 0.0) + itp
        if len(inter_totals) > 1:
            vals = list(inter_totals.values())
            s = sum(vals)
            sq = sum(v * v for v in vals)
            n_i = len(vals)
            fairness_score = float(min(1.0, (s * s) / (n_i * sq))) if sq > 0 else 1.0
        else:
            fairness_score = 1.0

        # Starvation: intersections where total throughput < 2% of expected
        expected_per_inter = max(n_steps * 0.5, 1.0)
        starvation_intersections = [
            iid for iid, v in inter_totals.items()
            if v < expected_per_inter * 0.02
        ]

        # Violations
        violations: List[str] = []
        if all_red_rate > 0.40:
            violations.append(f"ALL_RED abuse: {all_red_rate:.1%} of steps in ALL_RED")
        if phase_churn_rate > 0.50:
            violations.append(f"Phase churn: {phase_churn_rate:.1%} switch rate")
        if starvation_intersections:
            violations.append(f"Starvation at intersections: {starvation_intersections}")
        for ev in emergency_events:
            if not ev.get("served") and ev.get("latency_steps", 0) > 20:
                violations.append(
                    f"Emergency neglected: {_EMERG_NAMES.get(ev.get('etype',0),'?')} "
                    f"at I{ev.get('iid','?')} latency={ev.get('latency_steps','?')}s"
                )

        # Lessons
        lessons: List[str] = []
        if all_red_rate > 0.30:
            lessons.append("Avoid ALL_RED — it starves all lanes and wastes green time.")
        if phase_churn_rate > 0.40:
            lessons.append("Hold phases longer — switching too fast incurs penalties.")
        if emergency_events:
            unserved = [e for e in emergency_events if not e.get("served")]
            if unserved:
                lessons.append(
                    f"{len(unserved)} emergency(ies) unserved. Always route to emergency lane."
                )
            served = [e for e in emergency_events if e.get("served")]
            if served:
                avg_lat = _safe_mean([e.get("latency_steps", 0) for e in served])
                if avg_lat > 8:
                    lessons.append(f"Emergency latency high ({avg_lat:.1f} steps). Act faster.")
        if starvation_intersections:
            lessons.append("Some intersections had near-zero throughput. Distribute green time fairly.")
        if not lessons:
            lessons.append("Performance within expected range. Maintain current strategy.")

        score_breakdown: Dict[str, float] = {
            "throughput_per_step": round(total_throughput / max(n_steps, 1), 3),
            "avg_wait": round(avg_wait, 3),
            "spillback_mean_rate": round(spillback_summary["mean_rate"], 3),
            "fairness": round(fairness_score, 3),
            "phase_churn_rate": round(phase_churn_rate, 3),
            "all_red_rate": round(all_red_rate, 3),
        }

        return EpisodeFeedback(
            n_steps=n_steps,
            avg_wait_per_lane=round(avg_wait, 4),
            total_throughput=total_throughput,
            throughput_per_step=round(total_throughput / max(n_steps, 1), 3),
            emergency_events=emergency_events,
            spillback_summary=spillback_summary,
            violations=violations,
            fairness_score=round(fairness_score, 4),
            starvation_intersections=starvation_intersections,
            phase_churn_rate=round(phase_churn_rate, 4),
            all_red_rate=round(all_red_rate, 4),
            best_step=best_step,
            worst_step=worst_step,
            lessons=lessons[:5],
            score_breakdown=score_breakdown,
        )

    # ------------------------------------------------------------------
    # Action parsing / validation
    # ------------------------------------------------------------------

    def _parse_action(self, action: Any) -> TrafficAction:
        n = self.cfg.n_intersections
        if isinstance(action, TrafficAction):
            return action
        if isinstance(action, int):
            return TrafficAction.from_flat_int(action, n)
        if isinstance(action, (list, tuple, np.ndarray)):
            phases = [int(a) for a in action]
            # Validate each phase index — out-of-bound → hold (-1)
            valid = [p if 0 <= p < N_PHASES else -1 for p in phases]
            # Pad/truncate to n_intersections
            valid = (valid + [-1] * n)[:n]
            return TrafficAction(phase_indices=valid)
        # Unknown type → noop
        return TrafficAction.noop(n)

    def _resolve_emergency_override(self, action: TrafficAction) -> List[int]:
        """Apply emergency override if requested; return per-intersection phase list."""
        n = self.cfg.n_intersections
        phases = (list(action.phase_indices) + [-1] * n)[:n]

        if action.emergency_override >= 0:
            override_idx = action.emergency_override
            if 0 <= override_idx < n:
                # Force ALL_RED first (index 2)
                phases[override_idx] = int(PhaseEnum.ALL_RED)

        return phases

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot(self, state: TrafficState) -> Dict:
        """Compact snapshot for trajectory recording.

        All metrics are PER-STEP (not cumulative) so graders can use
        statistics.mean() directly without delta computation.
        Extended with emergency_events for accurate grader consumption.
        """
        n_lanes_total = sum(len(i.lanes) for i in state.intersections)

        # Per-step throughput = sum of lane.throughput (reset each step by simulator)
        per_step_tp = sum(
            l.throughput for i in state.intersections for l in i.lanes
        )

        # Per-step avg queue across all lanes (proxy for per-step waiting)
        total_queue = sum(
            l.queue_length for i in state.intersections for l in i.lanes
        )
        per_step_avg_wait = total_queue / max(n_lanes_total, 1)

        return {
            "step": state.step,
            "global_throughput": per_step_tp,        # vehicles discharged THIS step
            "global_avg_wait": per_step_avg_wait,    # avg queue length THIS step
            "phase_switches": state.phase_switches,
            # Accurate emergency event log from simulator
            "emergency_events": self._sim.get_served_emergency_events(),
            "intersections": [
                {
                    "id": i.intersection_id,
                    "phase": i.current_phase.value,
                    "emergency": i.emergency_active.value,
                    "emergency_lane": i.emergency_lane,
                    "weather": i.weather.value,
                    "queues": [l.queue_length for l in i.lanes],
                    "spillback": i.spillback_count,
                    # per-step throughput for this intersection
                    "throughput": sum(l.throughput for l in i.lanes),
                    "all_red_steps": i.all_red_steps,
                }
                for i in state.intersections
            ],
        }

    @property
    def trajectory(self) -> List[Dict]:
        """Full episode trajectory (for graders and analytics)."""
        return self._trajectory

    @property
    def n_intersections(self) -> int:
        return self.cfg.n_intersections

    @property
    def action_space_size(self) -> int:
        """Total number of discrete actions (N_PHASES ^ n_intersections)."""
        return N_PHASES ** self.cfg.n_intersections

    def action_space_description(self) -> str:
        return (
            f"Discrete({self.action_space_size}) — "
            f"{N_PHASES} phases per intersection × {self.cfg.n_intersections} intersections. "
            "Phase encoding: 0=NS_GREEN, 1=EW_GREEN, 2=ALL_RED"
        )

    def observation_space_description(self) -> str:
        fs = self.cfg.frame_stack
        H, W = self.cfg.image_size
        n = self.cfg.n_intersections
        return (
            f"frames: uint8 ({fs}, {H}, {W}, 3)  |  "
            f"metadata: float32 ({n}, 11)  |  "
            "metadata features: [q0,q1,q2,q3, phase, phase_timer, yellow_rem, "
            "emerg_type, emerg_lane, weather, spillback]"
        )
