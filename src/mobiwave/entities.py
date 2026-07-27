from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Vehicle:
    vehicle_id: int
    zone: int
    status: str = "idle"
    target_zone: Optional[int] = None
    remaining_time: int = 0
    pickup_remaining_time: int = 0
    active_request_id: Optional[int] = None
    idle_stay_steps: int = 0


@dataclass
class Request:
    request_id: int
    origin: int
    destination: int
    created_at: int
    fare: float


@dataclass
class DispatchDecision:
    assignments: Dict[int, int]
    rebalances: Dict[int, int]


@dataclass
class StepResult:
    time: int
    reward: float
    vehicle_rewards: Dict[int, float]
    served: int
    canceled: int
    new_requests: int
    open_requests: int
    net_reward: float = 0.0
    vehicle_net_rewards: Dict[int, float] = field(default_factory=dict)
    canceled_by_zone: Dict[int, int] = field(default_factory=dict)
