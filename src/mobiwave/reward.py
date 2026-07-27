from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np


REBALANCE_REWARD_COMPONENTS = (
    "deficit_bonus",
    "pressure_improvement",
    "open_request_bonus",
    "idle_spread_bonus",
    "movement_cost",
    "target_idle_penalty",
    "idle_cluster_penalty",
    "stay_streak_penalty",
    "vehicle_move_cost",
    "passenger_trip_revenue",
    "hold_penalty",
    "invalid_action_penalty",
)

DEFAULT_REBALANCE_REWARD_COMPONENTS = REBALANCE_REWARD_COMPONENTS


@dataclass(frozen=True)
class RebalanceRewardResult:
    reward: float
    components: dict[str, float]
    diagnostics: dict[str, float | int]
    is_move: bool
    next_stay_steps: int


def normalize_reward_components(components: Iterable[str] | str | None) -> tuple[str, ...]:
    if components is None:
        return DEFAULT_REBALANCE_REWARD_COMPONENTS
    if isinstance(components, str):
        raw = [part.strip() for part in components.replace(",", " ").split()]
    else:
        raw = [str(part).strip() for part in components]

    selected: list[str] = []
    for component in raw:
        if not component:
            continue
        if component == "passenger_trip_cost":
            component = "passenger_trip_revenue"
        if component == "all":
            return DEFAULT_REBALANCE_REWARD_COMPONENTS
        if component == "none":
            continue
        if component not in REBALANCE_REWARD_COMPONENTS:
            valid = ", ".join(REBALANCE_REWARD_COMPONENTS)
            raise ValueError(f"Unknown reward component '{component}'. Valid components: {valid}")
        if component not in selected:
            selected.append(component)
    return tuple(selected)


def compute_rebalance_reward(
    *,
    config,
    network,
    origin_zone: int,
    target_zone: int,
    travel_time: int,
    zone_open: np.ndarray,
    zone_idle: np.ndarray,
    expected: np.ndarray,
    idle_stay_steps: int,
    invalid_action: bool,
) -> RebalanceRewardResult:
    enabled = set(normalize_reward_components(config.reward_components))
    is_move = target_zone != origin_zone
    next_stay_steps = idle_stay_steps + 1 if not is_move else 0
    grid_distance = network.hex_distance(origin_zone, target_zone) if is_move else 0

    radius = max(0, config.rebalance_radius)
    target_neighbors = network.neighbors(target_zone, radius=radius)
    origin_neighbors = network.neighbors(origin_zone, radius=radius)
    target_pressure = (
        float(np.sum(zone_open[target_neighbors]))
        + float(np.sum(expected[target_neighbors]))
        - float(np.sum(zone_idle[target_neighbors]))
    )
    origin_pressure = (
        float(np.sum(zone_open[origin_neighbors]))
        + float(np.sum(expected[origin_neighbors]))
        - float(np.sum(zone_idle[origin_neighbors]))
    )

    state_radius = max(0, config.state_hop_radius)
    origin_state_neighbors = network.neighbors(origin_zone, radius=state_radius)
    target_state_neighbors = network.neighbors(target_zone, radius=state_radius)
    origin_local_demand = float(np.sum(zone_open[origin_state_neighbors])) + float(
        np.sum(expected[origin_state_neighbors])
    )
    target_local_demand = float(np.sum(zone_open[target_state_neighbors])) + float(
        np.sum(expected[target_state_neighbors])
    )
    origin_local_idle = float(np.sum(zone_idle[origin_state_neighbors]))
    target_local_idle = float(np.sum(zone_idle[target_state_neighbors]))
    origin_idle_after_departure = max(0.0, origin_local_idle - 1.0)
    target_idle_after_departure = (
        max(0.0, target_local_idle - 1.0)
        if origin_zone in target_state_neighbors
        else target_local_idle
    )
    origin_idle_surplus = max(0.0, origin_idle_after_departure - origin_local_demand)
    target_idle_surplus = max(0.0, target_idle_after_departure - target_local_demand)

    improvement = target_pressure - origin_pressure
    target_deficit = max(0.0, target_pressure)
    target_open_count = float(np.sum(zone_open[target_neighbors]))
    target_idle_count = float(np.sum(zone_idle[target_neighbors]))
    action_zone_idle_count = (
        float(zone_idle[target_zone]) + 1.0
        if is_move
        else float(zone_idle[origin_zone])
    )

    movement_cost = (
        config.rebalance_penalty * travel_time
        + (config.rebalance_move_fixed_cost if is_move else 0.0)
    )
    target_idle_penalty = (
        config.rebalance_target_idle_penalty * target_idle_count
        if is_move
        else 0.0
    )
    idle_cluster_penalty = config.idle_cluster_penalty * max(
        0.0,
        action_zone_idle_count - float(config.idle_cluster_free_threshold),
    )
    stay_streak_penalty = 0.0
    if not is_move:
        stay_exponent = min(10.0, config.stay_streak_growth * max(0, int((next_stay_steps - 1)/10)))
        stay_streak_penalty = min(
            config.stay_streak_penalty_cap,
            config.stay_streak_penalty * math.expm1(stay_exponent),
        )
    vehicle_move_cost = config.vehicle_move_cost_per_grid * grid_distance

    components = {
        "deficit_bonus": config.rebalance_deficit_weight * target_deficit,
        "pressure_improvement": (
            config.rebalance_improvement_weight * max(0.0, improvement - config.rebalance_move_threshold)
            - config.rebalance_worsen_penalty * max(0.0, config.rebalance_move_threshold - improvement)
            if is_move
            else 0.0
        ),
        "open_request_bonus": config.rebalance_open_request_weight * target_open_count,
        "idle_spread_bonus": (
            config.rebalance_idle_spread_weight * max(0.0, origin_idle_surplus - 0.5 * target_idle_surplus)
            if is_move
            else 0.0
        ),
        "movement_cost": -movement_cost,
        "target_idle_penalty": -target_idle_penalty,
        "idle_cluster_penalty": -idle_cluster_penalty,
        "stay_streak_penalty": -stay_streak_penalty,
        "vehicle_move_cost": -vehicle_move_cost,
        "passenger_trip_revenue": 0.0,
        "hold_penalty": (
            -config.idle_penalty * (1.0 + max(0.0, -origin_pressure))
            if not is_move
            else 0.0
        ),
        "invalid_action_penalty": -config.invalid_action_penalty if invalid_action else 0.0,
    }
    reward = sum(value for name, value in components.items() if name in enabled)
    diagnostics = {
        "target_pressure": target_pressure,
        "origin_pressure": origin_pressure,
        "pressure_improvement": improvement,
        "movement_cost": movement_cost,
        "grid_distance": grid_distance,
        "vehicle_move_cost": vehicle_move_cost,
        "passenger_trip_revenue": 0.0,
        "target_idle_penalty": target_idle_penalty,
        "idle_cluster_penalty": idle_cluster_penalty,
        "stay_streak_penalty": stay_streak_penalty,
        "idle_stay_steps": next_stay_steps,
    }
    return RebalanceRewardResult(
        reward=reward,
        components=components,
        diagnostics=diagnostics,
        is_move=is_move,
        next_stay_steps=next_stay_steps,
    )
