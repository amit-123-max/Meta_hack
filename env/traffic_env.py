"""Main TrafficSignalEnv — OpenEnv-compliant environment class."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.env_config import EnvConfig
from env.observation import ObservationBuilder
from env.reward import compute_reward
from env.schemas import (
    PhaseEnum,
    TrafficAction,
    TrafficObservation,
    TrafficReward,
    TrafficState,
)
from env.simulator import TrafficSimulator

# Number of discrete phases the agent can choose per intersection
N_PHASES = 3  # NS_GREEN, EW_GREEN, ALL_RED


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

    Episode
    -------
    An episode ends after SimConfig.max_steps steps (done=True).
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._sim = TrafficSimulator(cfg)
        self._obs_builder = ObservationBuilder(cfg)
        self._prev_state: Optional[TrafficState] = None
        self._trajectory: List[Dict] = []   # for graders / replay
        self._episode_reward: float = 0.0

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
        self._trajectory = []
        self._episode_reward = 0.0

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

        # Build observation
        obs = self._obs_builder.build(state)

        done = state.done

        info: Dict[str, Any] = {
            "step": state.step,
            "reward_breakdown": rew_obj.to_dict(),
            "global_throughput": state.global_throughput,
            "global_avg_wait": state.global_avg_wait,
            "phase_switches": state.phase_switches,
            "emergency_delays": list(state.episode_emergency_delays),
            "episode_reward": self._episode_reward,
        }

        # Record trajectory for graders / replay
        self._trajectory.append({
            "step": state.step,
            "action": parsed_action,
            "reward": reward,
            "reward_breakdown": rew_obj.to_dict(),
            "state_snapshot": self._snapshot(state),
        })

        self._prev_state = state
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
