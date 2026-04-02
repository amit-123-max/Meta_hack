"""Vehicle and EmergencyVehicle models — typed entities used in simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from env.schemas import EmergencyType


@dataclass
class Vehicle:
    """Represents a single vehicle in a lane queue."""
    vehicle_id: int
    arrival_step: int
    wait_steps: int = 0

    def tick(self) -> None:
        """Increment wait counter each step."""
        self.wait_steps += 1

    @property
    def wait_time(self) -> float:
        """Approximate waiting time in seconds (1 step = 1 second)."""
        return float(self.wait_steps)


@dataclass
class EmergencyVehicle(Vehicle):
    """An emergency vehicle with a priority type and active flag."""
    etype: EmergencyType = EmergencyType.AMBULANCE
    responded: bool = False        # True once it has had a green phase
    response_delay: float = 0.0   # Steps waited before first green

    # Priority hierarchy: ambulance > fire > police > normal
    PRIORITY_ORDER = {
        EmergencyType.AMBULANCE: 3,
        EmergencyType.FIRE:      2,
        EmergencyType.POLICE:    1,
        EmergencyType.NONE:      0,
    }

    @property
    def priority(self) -> int:
        return self.PRIORITY_ORDER.get(self.etype, 0)

    def mark_responded(self, current_step: int) -> None:
        if not self.responded:
            self.response_delay = float(current_step - self.arrival_step)
            self.responded = True

    def __lt__(self, other: "EmergencyVehicle") -> bool:
        """Higher priority vehicles sort first."""
        return self.priority > other.priority
