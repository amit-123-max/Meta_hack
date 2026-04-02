"""Core traffic simulator — deterministic, seed-based, configurable."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.env_config import EnvConfig, SimConfig
from env.schemas import (
    EmergencyType,
    IntersectionState,
    LaneState,
    PhaseEnum,
    TrafficState,
    WeatherCondition,
)

# ---------------------------------------------------------------------------
# Direction mapping
# ---------------------------------------------------------------------------
DIRECTIONS = ["N", "S", "E", "W"]

# Phase → which lane indices are green
# lane layout: [0=N, 1=S, 2=E, 3=W]
PHASE_GREEN_LANES: Dict[PhaseEnum, List[int]] = {
    PhaseEnum.NS_GREEN: [0, 1],   # North + South green
    PhaseEnum.EW_GREEN: [2, 3],   # East  + West  green
    PhaseEnum.ALL_RED:  [],        # No lane green
}


# ---------------------------------------------------------------------------
# Internal per-lane representation
# ---------------------------------------------------------------------------

@dataclass
class _Lane:
    lane_id: int
    direction: str
    queue: int = 0
    cumulative_wait: float = 0.0
    throughput_this_step: int = 0
    emergency: EmergencyType = EmergencyType.NONE
    emergency_timer: int = 0       # steps until emergency vehicle clears
    starvation_timer: int = 0      # steps without any green


# ---------------------------------------------------------------------------
# Per-intersection internal state
# ---------------------------------------------------------------------------

@dataclass
class _Intersection:
    iid: int
    lanes: List[_Lane]
    phase: PhaseEnum = PhaseEnum.NS_GREEN
    phase_timer: int = 0
    yellow_remaining: int = 0
    next_phase: Optional[PhaseEnum] = None  # queued after yellow clears
    weather: WeatherCondition = WeatherCondition.CLEAR
    spillback_count: int = 0
    total_throughput: int = 0
    total_wait: float = 0.0
    phase_switches: int = 0

    @property
    def n_lanes(self) -> int:
        return len(self.lanes)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class TrafficSimulator:
    """Simulates traffic across one or more intersections.

    Responsibilities:
    - Stochastic vehicle arrivals (Poisson)
    - Phase / phase-timer management with yellow transitions
    - Emergency vehicle spawning and priority clearing
    - Congestion propagation / spillback between neighbouring intersections
    - Weather state machine
    - Exposes full state and per-intersection lane states
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.sim: SimConfig = cfg.sim
        self._rng = np.random.default_rng(self.sim.seed)
        self._py_rng = random.Random(self.sim.seed)
        self._intersections: List[_Intersection] = []
        self._step: int = 0
        self._total_phase_switches: int = 0
        self._emergency_delays: List[float] = []
        self._active_emergency_arrivals: List[Dict] = []  # tracking

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self._rng = np.random.default_rng(self.sim.seed)
        self._py_rng = random.Random(self.sim.seed)
        self._step = 0
        self._total_phase_switches = 0
        self._emergency_delays = []
        self._active_emergency_arrivals = []
        self._intersections = [
            self._make_intersection(i) for i in range(self.cfg.n_intersections)
        ]

    def step(self, phase_actions: List[int]) -> Dict:
        """Advance simulation one step.

        Args:
            phase_actions: list of phase indices (int or -1 for hold),
                           one per intersection.

        Returns:
            dict with per-intersection stats for reward computation.
        """
        self._step += 1
        stats = []

        # Pad/truncate actions to match number of intersections
        n = self.cfg.n_intersections
        actions = (list(phase_actions) + [-1] * n)[:n]

        for i, inter in enumerate(self._intersections):
            action = int(actions[i])
            stat = self._step_intersection(inter, action, i)
            stats.append(stat)

        # Congestion propagation between neighbours (medium / hard tasks)
        if self.cfg.n_intersections > 1:
            self._propagate_spillback()

        # Weather state machine
        self._update_weather()

        return {"intersections": stats, "step": self._step}

    def get_state(self) -> TrafficState:
        """Return full typed state snapshot."""
        inter_states = [self._export_intersection(i) for i in self._intersections]
        global_tp = sum(s.total_throughput for s in inter_states)
        total_wait = sum(s.total_wait for s in inter_states)
        n_lanes = self.cfg.n_intersections * self.cfg.lanes_per_intersection
        global_avg_wait = total_wait / max(n_lanes, 1)
        return TrafficState(
            step=self._step,
            intersections=inter_states,
            global_throughput=global_tp,
            global_avg_wait=global_avg_wait,
            episode_emergency_delays=list(self._emergency_delays),
            phase_switches=self._total_phase_switches,
            done=self._step >= self.sim.max_steps,
        )

    @property
    def step_count(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    # Intersection construction
    # ------------------------------------------------------------------

    def _make_intersection(self, iid: int) -> _Intersection:
        lanes = [
            _Lane(lane_id=j, direction=DIRECTIONS[j])
            for j in range(self.cfg.lanes_per_intersection)
        ]
        return _Intersection(iid=iid, lanes=lanes)

    # ------------------------------------------------------------------
    # Per-intersection step
    # ------------------------------------------------------------------

    def _step_intersection(
        self, inter: _Intersection, action: int, inter_idx: int
    ) -> Dict:
        # Reset per-step throughput
        for lane in inter.lanes:
            lane.throughput_this_step = 0

        # 1. Vehicle arrivals (Poisson)
        self._arrive_vehicles(inter)

        # 2. Emergency vehicle spawning
        self._maybe_spawn_emergency(inter)

        # 3. Apply action (phase transition)
        switched = self._apply_action(inter, action)
        if switched:
            self._total_phase_switches += 1
            inter.phase_switches += 1

        # 4. Yellow countdown
        if inter.yellow_remaining > 0:
            inter.yellow_remaining -= 1
            if inter.yellow_remaining == 0 and inter.next_phase is not None:
                inter.phase = inter.next_phase
                inter.next_phase = None
                inter.phase_timer = 0

        # 5. Service green lanes
        spillback_occurred = self._service_lanes(inter)
        if spillback_occurred:
            inter.spillback_count += 1

        # 6. Accumulate waiting (all non-served vehicles in queue)
        total_wait_this_step = 0.0
        for lane in inter.lanes:
            wait = float(lane.queue) * self.sim.dt
            lane.cumulative_wait += wait
            total_wait_this_step += wait

            # Starvation tracking
            green_lanes = PHASE_GREEN_LANES.get(inter.phase, [])
            if lane.lane_id in green_lanes:
                lane.starvation_timer = 0
            else:
                lane.starvation_timer += 1

        inter.total_wait += total_wait_this_step
        inter.phase_timer += 1

        # 7. Emergency timer countdown
        for lane in inter.lanes:
            if lane.emergency != EmergencyType.NONE and lane.emergency_timer > 0:
                lane.emergency_timer -= 1
                if lane.emergency_timer == 0:
                    # Emergency vehicle cleared — record delay
                    # Delay = steps it was queued waiting (starvation_timer proxy)
                    delay = float(lane.starvation_timer) * self.sim.dt
                    self._emergency_delays.append(delay)
                    lane.emergency = EmergencyType.NONE
                    lane.starvation_timer = 0

        return {
            "intersection_id": inter.iid,
            "throughput": sum(l.throughput_this_step for l in inter.lanes),
            "wait": total_wait_this_step,
            "spillback": spillback_occurred,
            "phase_switched": switched,
            "lanes": [
                {
                    "lane_id": l.lane_id,
                    "queue": l.queue,
                    "throughput": l.throughput_this_step,
                    "emergency": l.emergency,
                    "starvation": l.starvation_timer,
                }
                for l in inter.lanes
            ],
        }

    # ------------------------------------------------------------------
    # Vehicle arrivals
    # ------------------------------------------------------------------

    def _arrive_vehicles(self, inter: _Intersection) -> None:
        base = self.sim.arrival_rate_base
        noise = self.sim.arrival_rate_noise
        for lane in inter.lanes:
            lam = base + self._rng.uniform(-noise, noise)
            lam = max(0.0, lam)
            arrivals = int(self._rng.poisson(lam))
            lane.queue = min(lane.queue + arrivals, self.sim.max_queue_per_lane)

    # ------------------------------------------------------------------
    # Emergency spawning
    # ------------------------------------------------------------------

    def _maybe_spawn_emergency(self, inter: _Intersection) -> None:
        if self.sim.emergency_prob_per_step <= 0.0:
            return
        if self._rng.random() < self.sim.emergency_prob_per_step:
            # Pick a random lane that has no active emergency
            candidates = [
                l for l in inter.lanes if l.emergency == EmergencyType.NONE
            ]
            if not candidates:
                return
            lane = self._py_rng.choice(candidates)
            # Assign priority based on weighted random
            etype = self._py_rng.choices(
                [EmergencyType.POLICE, EmergencyType.FIRE, EmergencyType.AMBULANCE],
                weights=[3, 2, 1],
                k=1,
            )[0]
            lane.emergency = etype
            lane.emergency_timer = self.sim.emergency_clear_steps
            lane.queue = min(lane.queue + 1, self.sim.max_queue_per_lane)
            # Track emergency arrival for delay measurement
            self._active_emergency_arrivals.append({
                "arrival_step": self._step,
                "intersection": inter.iid,
                "lane": lane.lane_id,
                "etype": etype,
            })

    # ------------------------------------------------------------------
    # Phase action application
    # ------------------------------------------------------------------

    def _apply_action(self, inter: _Intersection, action: int) -> bool:
        """Apply agent's phase action. Returns True if phase switched."""
        # Invalid / hold action
        valid_phases = [PhaseEnum.NS_GREEN, PhaseEnum.EW_GREEN, PhaseEnum.ALL_RED]
        if action < 0 or action >= len(valid_phases):
            return False

        desired_phase = valid_phases[action]

        # Same phase — no switch
        if desired_phase == inter.phase and inter.yellow_remaining == 0:
            return False

        # Min phase duration enforcement (prevent oscillation)
        if inter.phase_timer < self.sim.phase_duration_min:
            return False

        # If in yellow, queue the new phase
        if inter.yellow_remaining > 0:
            inter.next_phase = desired_phase
            return False

        # Initiate phase switch with yellow transition
        inter.yellow_remaining = self.sim.yellow_duration
        inter.next_phase = desired_phase
        return True

    # ------------------------------------------------------------------
    # Lane servicing
    # ------------------------------------------------------------------

    def _service_lanes(self, inter: _Intersection) -> bool:
        """Release vehicles from green lanes. Returns True if spillback."""
        if inter.yellow_remaining > 0:
            return False  # no service during yellow

        green_lanes = PHASE_GREEN_LANES.get(inter.phase, [])
        spillback = False

        for lane in inter.lanes:
            if lane.lane_id not in green_lanes:
                continue
            released = min(lane.queue, self.sim.discharge_rate)
            lane.queue -= released
            lane.throughput_this_step += released
            inter.total_throughput += released

            # Check if downstream overflow would cause spillback
            if lane.queue >= int(self.sim.max_queue_per_lane * self.sim.spillback_threshold):
                spillback = True

        return spillback

    # ------------------------------------------------------------------
    # Congestion propagation
    # ------------------------------------------------------------------

    def _propagate_spillback(self) -> None:
        """Push overflow from saturated queues to upstream neighbours."""
        adj = self.cfg.adjacency or []
        for i, inter in enumerate(self._intersections):
            neighbours = adj[i] if i < len(adj) else []
            if not neighbours:
                continue
            for lane in inter.lanes:
                fill_ratio = lane.queue / self.sim.max_queue_per_lane
                if fill_ratio >= self.sim.spillback_threshold:
                    overflow = int(
                        (fill_ratio - self.sim.spillback_threshold)
                        * self.sim.max_queue_per_lane
                        * self.sim.propagation_fraction
                    )
                    if overflow <= 0:
                        continue
                    # Distribute to random neighbour's matching lane
                    nbr_idx = self._py_rng.choice(neighbours)
                    nbr = self._intersections[nbr_idx]
                    nbr_lane = nbr.lanes[lane.lane_id % len(nbr.lanes)]
                    nbr_lane.queue = min(
                        nbr_lane.queue + overflow, self.sim.max_queue_per_lane
                    )

    # ------------------------------------------------------------------
    # Weather state machine
    # ------------------------------------------------------------------

    def _update_weather(self) -> None:
        if self.sim.weather_change_prob <= 0.0:
            return
        for inter in self._intersections:
            if self._rng.random() < self.sim.weather_change_prob:
                inter.weather = WeatherCondition(
                    int(self._rng.integers(0, len(WeatherCondition)))
                )

    # ------------------------------------------------------------------
    # State export helpers
    # ------------------------------------------------------------------

    def _export_intersection(self, inter: _Intersection) -> IntersectionState:
        emergency_active = EmergencyType.NONE
        emergency_lane = -1
        for lane in inter.lanes:
            if lane.emergency.value > emergency_active.value:
                emergency_active = lane.emergency
                emergency_lane = lane.lane_id

        lane_states = [
            LaneState(
                lane_id=l.lane_id,
                direction=l.direction,
                queue_length=l.queue,
                throughput=l.throughput_this_step,
                wait_time=l.cumulative_wait,
                is_green=(
                    l.lane_id in PHASE_GREEN_LANES.get(inter.phase, [])
                    and inter.yellow_remaining == 0
                ),
                emergency=l.emergency,
                is_occluded=(
                    self._rng.random() < self.sim.occlusion_prob
                ),
            )
            for l in inter.lanes
        ]

        return IntersectionState(
            intersection_id=inter.iid,
            lanes=lane_states,
            current_phase=inter.phase,
            phase_timer=inter.phase_timer,
            yellow_remaining=inter.yellow_remaining,
            emergency_active=emergency_active,
            emergency_lane=emergency_lane,
            weather=inter.weather,
            total_throughput=inter.total_throughput,
            total_wait=inter.total_wait,
            spillback_count=inter.spillback_count,
        )

    # ------------------------------------------------------------------
    # Raw accessor (for observation builder)
    # ------------------------------------------------------------------

    def get_intersections_raw(self) -> List[_Intersection]:
        return self._intersections

    def get_emergency_delays(self) -> List[float]:
        return list(self._emergency_delays)
