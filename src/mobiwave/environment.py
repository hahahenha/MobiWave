from __future__ import annotations

from typing import Any, Dict, List
import math
import copy
import time
import numpy as np

from .config import SimulationConfig
from .demand import DemandGenerator
from .entities import DispatchDecision, Request, StepResult, Vehicle
from .network import GridNetwork
from .history import CausalDemandHistory
from .reward import compute_rebalance_reward
from .scenarios import DriftScenario


class DispatchEnv:
    """Pure Python zone-level ride-hailing dispatch simulator."""

    def __init__(
        self,
        config: SimulationConfig,
        scenario: DriftScenario | None = None,
        *,
        metric_start_time: int = 0,
        record_detailed_logs: bool = True,
    ) -> None:
        self.config = config
        # Scenario ground truth is deliberately private: SimulationConfig is
        # included in policy snapshots, whereas onset/recovery are evaluator-only.
        self.scenario = scenario or DriftScenario.for_horizon("no_drift", config.horizon)
        self.metric_start_time = int(metric_start_time)
        self.record_detailed_logs = bool(record_detailed_logs)
        if not 0 <= self.metric_start_time < config.horizon:
            raise ValueError("metric_start_time must be within the simulation horizon")
        self.network = GridNetwork(config.grid_rows, config.grid_cols)
        self.rng = np.random.default_rng(np.random.SeedSequence([config.seed, 7300]))
        self.demand = DemandGenerator(config, self.network, self.rng, scenario=self.scenario)
        self.demand_history = None
        self._bound_policy_id: int | None = None
        self._bound_policy = None
        self.time = 0
        self.vehicles: Dict[int, Vehicle] = {}
        self.open_requests: Dict[int, Request] = {}
        self.metrics: Dict[str, float] = {}
        self.summary_log: List[Dict[str, Any]] = []
        self.vehicle_log: List[Dict[str, Any]] = []
        self.zone_log: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self._current_new_requests: List[Request] = []
        self._last_canceled_by_zone: Dict[int, int] = {}
        self._prepared_decision: tuple[Dict[str, Any], int, int] | None = None
        self._historical_od_counts = np.zeros(
            (self.network.zone_count, self.network.zone_count),
            dtype=np.float64,
        )
        self._freeze_historical_od = False
        self._metric_window_started = self.metric_start_time == 0
        self.reset()

    def reset(self, seed: int | None = None) -> Dict:
        history_config = self.config.with_updates(seed=seed) if seed is not None else self.config
        self.rng = np.random.default_rng(
            np.random.SeedSequence([int(history_config.seed), 7300])
        )
        self.demand = DemandGenerator(
            history_config,
            self.network,
            self.rng,
            scenario=self.scenario,
        )
        self.demand_history = CausalDemandHistory(history_config, self.network)
        self.time = 0
        self.open_requests = {}
        self.vehicles = {}
        self.metrics = self._empty_metrics()
        self._metric_window_started = self.metric_start_time == 0
        self.summary_log = []
        self.vehicle_log = []
        self.zone_log = []
        self.event_log = []
        self._current_new_requests = []
        self._last_canceled_by_zone = {}
        self._prepared_decision = None
        self._historical_od_counts = np.zeros(
            (self.network.zone_count, self.network.zone_count),
            dtype=np.float64,
        )
        self._freeze_historical_od = False
        self._bound_policy_id = None
        self._bound_policy = None
        self._init_fleet()
        self._supply_vehicle_order = self._build_supply_vehicle_order(history_config.seed)
        self._sync_supply_availability()
        return self.snapshot()

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "served": 0,
            "canceled": 0,
            "generated": 0,
            "revenue": 0.0,
            "reward": 0.0,
            "net_reward": 0.0,
            "vehicle_move_cost_total": 0.0,
            "passenger_trip_revenue_total": 0.0,
            "operating_cost_total": 0.0,
            "pickup_time_total": 0.0,
            "wait_time_total": 0.0,
            "rebalance_time_total": 0.0,
            "empty_vehicle_steps": 0.0,
            "total_active_vehicle_steps": 0.0,
            "offline_vehicle_steps": 0.0,
            "max_offline_vehicles": 0.0,
            "demand_history_observation_time_total": 0.0,
            "dispatch_state_time_total": 0.0,
            "policy_decision_time_total": 0.0,
            "policy_decision_count": 0.0,
        }

    def _start_metric_window(self) -> None:
        if self._metric_window_started or self.time != self.metric_start_time:
            return
        self.metrics = self._empty_metrics()
        self.summary_log = []
        self.vehicle_log = []
        self.zone_log = []
        self.event_log = []
        self._metric_window_started = True

    @property
    def done(self) -> bool:
        return self.time >= self.config.horizon

    def _init_fleet(self) -> None:
        zone_weights = np.full(self.network.zone_count, 0.5, dtype=float)
        center = self.network.to_zone(self.config.grid_rows // 2, self.config.grid_cols // 2)
        for zone in range(self.network.zone_count):
            zone_weights[zone] += 1.0 / (1.0 + self.network.manhattan(center, zone))
        zone_weights /= zone_weights.sum()
        zones = self.rng.choice(self.network.zone_count, size=self.config.fleet_size, p=zone_weights)
        for vehicle_id, zone in enumerate(zones):
            self.vehicles[vehicle_id] = Vehicle(vehicle_id=vehicle_id, zone=int(zone))

    def snapshot(self) -> Dict:
        zone_open = np.zeros(self.network.zone_count, dtype=int)
        zone_backlog = np.zeros(self.network.zone_count, dtype=int)
        current_request_ids = {
            int(request.request_id) for request in self._current_new_requests
        }
        for request in self.open_requests.values():
            zone_open[request.origin] += 1
            if int(request.request_id) not in current_request_ids:
                zone_backlog[request.origin] += 1
        zone_idle = np.zeros(self.network.zone_count, dtype=int)
        for vehicle in self.vehicles.values():
            if vehicle.status == "idle":
                zone_idle[vehicle.zone] += 1
        expected = self.demand_history.expected_counts(self.time, self.config.demand_window)
        zone_new = np.zeros(self.network.zone_count, dtype=int)
        for request in self._current_new_requests:
            zone_new[request.origin] += 1
        feasible_destinations = {
            zone: tuple(
                destination
                for destination in sorted(
                    {
                        int(self.network.move(zone, action))
                        for action in self.network.valid_actions(zone)
                    }
                )
                if (
                    destination == zone
                    or self.scenario.edge_available(
                        self.network,
                        zone,
                        destination,
                        self.time,
                    )
                )
            )
            for zone in range(self.network.zone_count)
        }
        edge_travel_times = {
            (source, destination): self._actual_travel_time(source, destination)
            for source, destinations in feasible_destinations.items()
            for destination in destinations
        }
        free_flow_travel_times = {
            (source, destination): self.network.free_flow_time(
                source,
                destination,
            )
            for source, destinations in feasible_destinations.items()
            for destination in destinations
        }
        edge_graph_weights = self._graph_edge_weights(
            feasible_destinations,
            free_flow_travel_times,
        )
        return {
            "time": self.time,
            "config": self.config,
            "network": self.network,
            "vehicles": copy.deepcopy(self.vehicles),
            "requests": copy.deepcopy(self.open_requests),
            "zone_open_requests": zone_open,
            "zone_backlog_requests": zone_backlog,
            "zone_idle_vehicles": zone_idle,
            "zone_new_requests": zone_new,
            "new_requests": tuple(copy.deepcopy(self._current_new_requests)),
            "expected_demand": expected,
            "feasible_destinations": feasible_destinations,
            "edge_travel_times": edge_travel_times,
            "edge_graph_weights": edge_graph_weights,
            "graph_version": self.scenario.graph_version(self.network, self.time),
            # These requests expired before the current decision.  Exposing
            # only observed counts lets a learner credit the preceding action
            # without revealing evaluator-only drift boundaries.
            "canceled_before": int(sum(self._last_canceled_by_zone.values())),
            "canceled_by_zone": dict(self._last_canceled_by_zone),
            "metrics": dict(self.metrics),
        }

    def step(self, policy) -> StepResult:
        if self.done:
            return StepResult(
                time=self.time,
                reward=0.0,
                vehicle_rewards={},
                served=0,
                canceled=0,
                new_requests=0,
                open_requests=len(self.open_requests),
            )

        self._bind_policy(policy)
        snapshot_before, canceled_before, new_request_count = (
            self.advance_to_decision_boundary()
        )
        decision_started_at = time.perf_counter()
        decision = policy.decide(snapshot_before)
        self.metrics["policy_decision_time_total"] += time.perf_counter() - decision_started_at
        self.metrics["policy_decision_count"] += 1
        result = self._apply_decision(
            decision,
            canceled_before,
            new_request_count,
            snapshot_before,
        )
        self.metrics["reward"] += result.reward
        self.metrics["net_reward"] += result.net_reward
        self._record_logs(
            new_requests=new_request_count,
            step_reward=result.reward,
            step_served=result.served,
            step_canceled=result.canceled,
        )

        self.time += 1
        self._current_new_requests = []
        self._prepared_decision = None
        if hasattr(policy, "observe_transition"):
            policy.observe_transition(snapshot_before, decision, result)
        return result

    def advance_to_decision_boundary(self) -> tuple[Dict[str, Any], int, int]:
        """Advance exogenous/pre-action events once, without executing U_t."""

        if self.done:
            raise RuntimeError("cannot prepare a decision after the horizon")
        if self._prepared_decision is not None:
            return self._prepared_decision
        self._start_metric_window()
        self._record_scenario_marker()
        self._sync_supply_availability()
        self._advance_vehicles()
        # A selected vehicle that completes a trip during a supply shock enters
        # downtime immediately instead of becoming dispatchable for this step.
        self._sync_supply_availability()
        new_requests = self.demand.sample_requests(self.time)
        self._current_new_requests = list(new_requests)
        for request in new_requests:
            self.open_requests[request.request_id] = request
            if not self._freeze_historical_od:
                self._historical_od_counts[
                    int(request.origin),
                    int(request.destination),
                ] += 1.0
        history_started_at = time.perf_counter()
        self.demand_history.observe(self.time, new_requests)
        self.metrics["demand_history_observation_time_total"] += (
            time.perf_counter() - history_started_at
        )
        self.metrics["generated"] += len(new_requests)

        canceled_before = self._expire_requests()
        state_started_at = time.perf_counter()
        snapshot_before = self.snapshot()
        self.metrics["dispatch_state_time_total"] += time.perf_counter() - state_started_at
        self._prepared_decision = (
            snapshot_before,
            int(canceled_before),
            len(new_requests),
        )
        return self._prepared_decision

    def run(self, policy) -> Dict[str, Any]:
        self._bind_policy(policy)
        while not self.done:
            self.step(policy)
        return self.summary()

    def _bind_policy(self, policy) -> None:
        policy_id = id(policy)
        if self._bound_policy_id == policy_id:
            return
        if hasattr(policy, "bind_environment"):
            policy.bind_environment(self)
        self._bound_policy_id = policy_id
        self._bound_policy = policy

    def export_replay_state(self) -> Dict[str, Any]:
        """Return an opaque, pre-step checkpoint for paired suffix replay."""

        if self._prepared_decision is not None:
            raise RuntimeError(
                "replay checkpoints must be captured before decision-boundary preparation"
            )
        return {
            "time": int(self.time),
            "vehicles": copy.deepcopy(self.vehicles),
            "open_requests": copy.deepcopy(self.open_requests),
            "metrics": copy.deepcopy(self.metrics),
            "demand_history": copy.deepcopy(self.demand_history),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "supply_vehicle_order": tuple(self._supply_vehicle_order),
            "current_new_requests": copy.deepcopy(self._current_new_requests),
            "last_canceled_by_zone": dict(self._last_canceled_by_zone),
            "historical_od_counts": self._historical_od_counts.copy(),
            "freeze_historical_od": self._freeze_historical_od,
        }

    def import_replay_state(self, state: Dict[str, Any]) -> None:
        """Restore a checkpoint into an isolated environment instance."""

        required = {
            "time",
            "vehicles",
            "open_requests",
            "metrics",
            "demand_history",
            "rng_state",
            "supply_vehicle_order",
            "current_new_requests",
            "historical_od_counts",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"Replay checkpoint is missing: {sorted(missing)}")
        self.time = int(state["time"])
        self.vehicles = copy.deepcopy(state["vehicles"])
        self.open_requests = copy.deepcopy(state["open_requests"])
        self.metrics = copy.deepcopy(state["metrics"])
        self.demand_history = copy.deepcopy(state["demand_history"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._supply_vehicle_order = tuple(state["supply_vehicle_order"])
        self._current_new_requests = copy.deepcopy(state["current_new_requests"])
        self._prepared_decision = None
        self._last_canceled_by_zone = {
            int(zone): int(count)
            for zone, count in state.get("last_canceled_by_zone", {}).items()
        }
        self._historical_od_counts = np.asarray(
            state["historical_od_counts"],
            dtype=np.float64,
        ).copy()
        self._freeze_historical_od = bool(
            state.get("freeze_historical_od", False)
        )
        self.summary_log = []
        self.vehicle_log = []
        self.zone_log = []
        self.event_log = []
        self._bound_policy_id = None
        self._bound_policy = None

    def historical_od_counts(self) -> np.ndarray:
        """Return a detached copy of the causal historical OD counts."""

        return self._historical_od_counts.copy()

    def set_historical_od_prior(self, counts: np.ndarray) -> None:
        """Install the offline OD prior before a fresh test stream starts."""

        if self.time != 0 or self._prepared_decision is not None:
            raise RuntimeError(
                "historical OD prior can only be installed before the first step"
            )
        values = np.asarray(counts, dtype=np.float64)
        expected = (self.network.zone_count, self.network.zone_count)
        if (
            values.shape != expected
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ValueError(
                f"historical OD prior must be a finite nonnegative {expected} matrix"
            )
        self._historical_od_counts = values.copy()
        self._freeze_historical_od = True

    def summary(self) -> Dict[str, Any]:
        served = max(1.0, self.metrics["served"])
        fleet = max(1.0, float(self.config.fleet_size))
        total_cost = float(self.metrics["operating_cost_total"])
        total_revenue = float(self.metrics["revenue"])
        total_active_vehicle_steps = float(self.metrics["total_active_vehicle_steps"])
        empty_loaded_rate = (
            float(self.metrics["empty_vehicle_steps"]) / total_active_vehicle_steps
            if total_active_vehicle_steps > 0.0
            else 0.0
        )
        decision_time = float(self.metrics["policy_decision_time_total"])
        state_time = float(self.metrics["dispatch_state_time_total"])
        decision_count = int(self.metrics["policy_decision_count"])
        counts = self._fleet_counts()
        return {
            "served": int(self.metrics["served"]),
            "canceled": int(self.metrics["canceled"]),
            "generated": int(self.metrics["generated"]),
            "empty_loaded_rate": empty_loaded_rate,
            "evaluation_start_time": self.metric_start_time,
            "evaluation_steps": max(0, int(self.time) - self.metric_start_time),
            "empty_vehicle_steps": int(self.metrics["empty_vehicle_steps"]),
            "total_active_vehicle_steps": int(total_active_vehicle_steps),
            "revenue": round(self.metrics["revenue"], 3),
            "vehicle_move_cost": round(self.metrics["vehicle_move_cost_total"], 3),
            "passenger_trip_revenue": round(self.metrics["passenger_trip_revenue_total"], 3),
            "total_operating_cost": round(total_cost, 3),
            "profit": round(total_revenue - total_cost, 3),
            "avg_vehicle_revenue": total_revenue / fleet,
            "avg_vehicle_cost": total_cost / fleet,
            "avg_vehicle_profit": (total_revenue - total_cost) / fleet,
            "reward": round(self.metrics["reward"], 3),
            "net_reward": round(self.metrics["net_reward"], 3),
            "avg_pickup_time": self.metrics["pickup_time_total"] / served,
            "avg_wait_time": self.metrics["wait_time_total"] / served,
            "rebalance_time": self.metrics["rebalance_time_total"],
            "demand_history_observation_time_s": self.metrics["demand_history_observation_time_total"],
            "dispatch_state_time_s": state_time,
            "policy_decision_time_s": decision_time,
            "avg_policy_decision_latency_ms": (
                1000.0 * decision_time / decision_count if decision_count else 0.0
            ),
            "policy_decisions_per_second": (
                decision_count / decision_time if decision_time > 0.0 else 0.0
            ),
            "avg_dispatch_inference_latency_ms": (
                1000.0 * (state_time + decision_time) / decision_count
                if decision_count
                else 0.0
            ),
            "available_vehicles": int(self.config.fleet_size - counts["offline"]),
            "max_offline_vehicles": int(self.metrics["max_offline_vehicles"]),
            "scenario": self.scenario.family,
            "scenario_label": self.scenario.label,
            "scenario_onset": self.scenario.onset,
            "scenario_recovery": self.scenario.recovery,
            "scenario_intensity": self.scenario.intensity,
            "scenario_matched_to": self.scenario.matched_to,
            "scenario_is_control": self.scenario.evaluation_control,
        }

    def _advance_vehicles(self) -> None:
        for vehicle in self.vehicles.values():
            if vehicle.status in {"idle", "offline"}:
                continue
            vehicle.remaining_time -= 1
            if vehicle.status == "serving" and vehicle.pickup_remaining_time > 0:
                vehicle.pickup_remaining_time -= 1
            if vehicle.remaining_time <= 0:
                if vehicle.target_zone is not None:
                    vehicle.zone = vehicle.target_zone
                vehicle.status = "idle"
                vehicle.target_zone = None
                vehicle.active_request_id = None
                vehicle.remaining_time = 0
                vehicle.pickup_remaining_time = 0
                vehicle.idle_stay_steps = 0

    def _build_supply_vehicle_order(self, seed: int) -> tuple[int, ...]:
        vehicle_ids = np.arange(self.config.fleet_size, dtype=int)
        supply_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7302]))
        supply_rng.shuffle(vehicle_ids)
        return tuple(int(vehicle_id) for vehicle_id in vehicle_ids)

    def _unavailable_vehicle_ids(self) -> set[int]:
        fraction = self.scenario.unavailable_fraction(self.time)
        count = min(self.config.fleet_size, int(math.ceil(self.config.fleet_size * fraction)))
        return set(self._supply_vehicle_order[:count])

    def _sync_supply_availability(self) -> None:
        unavailable = self._unavailable_vehicle_ids()
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.status == "offline" and vehicle_id not in unavailable:
                vehicle.status = "idle"
                vehicle.idle_stay_steps = 0
            elif vehicle.status == "idle" and vehicle_id in unavailable:
                vehicle.status = "offline"
                vehicle.target_zone = None
                vehicle.active_request_id = None
                vehicle.remaining_time = 0
                vehicle.pickup_remaining_time = 0
                vehicle.idle_stay_steps = 0

    def _expire_requests(self) -> int:
        expired = [
            request_id
            for request_id, request in self.open_requests.items()
            if self.time - request.created_at > self.config.max_wait
        ]
        canceled_by_zone: Dict[int, int] = {}
        for request_id in expired:
            request = self.open_requests.pop(request_id)
            origin = int(request.origin)
            canceled_by_zone[origin] = canceled_by_zone.get(origin, 0) + 1
            self.metrics["canceled"] += 1
            self.event_log.append(
                {
                    "time": self.time,
                    "event": "cancel",
                    "request_id": request_id,
                    "origin": request.origin,
                    "destination": request.destination,
                    "vehicle_id": -1,
                    "reward": -self.config.cancel_penalty,
                }
            )
        self._last_canceled_by_zone = canceled_by_zone
        return len(expired)

    def _apply_decision(
        self,
        decision: DispatchDecision,
        canceled_before: int,
        new_request_count: int,
        snapshot_before: Dict,
    ) -> StepResult:
        reward = -self.config.cancel_penalty * canceled_before
        net_reward = -self.config.cancel_penalty * canceled_before
        vehicle_rewards: Dict[int, float] = {}
        vehicle_net_rewards: Dict[int, float] = {}
        served = 0
        assigned_vehicles = set()
        assigned_requests = set()

        for vehicle_id, request_id in decision.assignments.items():
            vehicle = self.vehicles.get(vehicle_id)
            request = self.open_requests.get(request_id)
            if vehicle is None or request is None:
                continue
            if vehicle.status != "idle" or vehicle_id in assigned_vehicles or request_id in assigned_requests:
                continue
            pickup = self._actual_travel_time(vehicle.zone, request.origin)
            trip = self._actual_travel_time(request.origin, request.destination)
            pickup_grid_distance = self.network.hex_distance(vehicle.zone, request.origin)
            passenger_grid_distance = self.network.hex_distance(request.origin, request.destination)
            pickup_vehicle_move_cost = self.config.vehicle_move_cost_per_grid * pickup_grid_distance
            loaded_vehicle_move_cost = self.config.vehicle_move_cost_per_grid * passenger_grid_distance
            vehicle_move_cost = pickup_vehicle_move_cost + loaded_vehicle_move_cost
            passenger_trip_revenue = self.config.passenger_trip_revenue_per_grid * passenger_grid_distance
            service_revenue = request.fare + passenger_trip_revenue
            operating_cost = vehicle_move_cost
            wait = max(0, self.time - request.created_at)
            vehicle.status = "serving"
            vehicle.target_zone = request.destination
            vehicle.remaining_time = pickup + trip
            vehicle.pickup_remaining_time = pickup
            vehicle.active_request_id = request.request_id
            vehicle.idle_stay_steps = 0
            self.open_requests.pop(request_id, None)
            assigned_vehicles.add(vehicle_id)
            assigned_requests.add(request_id)
            served += 1

            enabled_reward_components = set(self.config.reward_components)
            reward_vehicle_move_cost = (
                -vehicle_move_cost if "vehicle_move_cost" in enabled_reward_components else 0.0
            )
            reward_passenger_trip_revenue = (
                passenger_trip_revenue if "passenger_trip_revenue" in enabled_reward_components else 0.0
            )
            service_reward = (
                request.fare
                - self.config.pickup_penalty * pickup
                - self.config.travel_penalty * trip
                - 0.14 * wait
                + reward_vehicle_move_cost
                + reward_passenger_trip_revenue
            )
            vehicle_rewards[vehicle_id] = service_reward
            service_net_reward = (
                service_revenue
                # Paper objective: fare revenue minus *empty* travel,
                # passenger waiting, and cancellation costs.  Loaded-trip
                # operating cost remains available in the separate profit
                # metric but is not subtracted again from r_t.
                - pickup_vehicle_move_cost
                - self.config.pickup_penalty * pickup
                - 0.14 * wait
            )
            vehicle_net_rewards[vehicle_id] = service_net_reward
            reward += service_reward
            net_reward += service_net_reward
            self.metrics["served"] += 1
            self.metrics["revenue"] += service_revenue
            self.metrics["vehicle_move_cost_total"] += vehicle_move_cost
            self.metrics["passenger_trip_revenue_total"] += passenger_trip_revenue
            self.metrics["operating_cost_total"] += operating_cost
            self.metrics["pickup_time_total"] += pickup
            self.metrics["wait_time_total"] += wait + pickup
            self.event_log.append(
                {
                    "time": self.time,
                    "event": "serve",
                    "vehicle_id": vehicle_id,
                    "request_id": request_id,
                    "origin": request.origin,
                    "destination": request.destination,
                    "pickup_time": pickup,
                    "trip_time": trip,
                    "pickup_grid_distance": pickup_grid_distance,
                    "passenger_grid_distance": passenger_grid_distance,
                    "wait_time": wait,
                    "base_fare": round(request.fare, 3),
                    "fare": round(service_revenue, 3),
                    "reward_components": ",".join(self.config.reward_components),
                    "pickup_vehicle_move_cost": round(pickup_vehicle_move_cost, 3),
                    "loaded_vehicle_move_cost": round(loaded_vehicle_move_cost, 3),
                    "vehicle_move_cost": round(vehicle_move_cost, 3),
                    "passenger_trip_revenue": round(passenger_trip_revenue, 3),
                    "operating_cost": round(operating_cost, 3),
                    "reward_vehicle_move_cost": round(-vehicle_move_cost, 3),
                    "reward_passenger_trip_revenue": round(reward_passenger_trip_revenue, 3),
                    "reward": round(service_reward, 3),
                }
            )

        expected = snapshot_before["expected_demand"]
        zone_idle = snapshot_before["zone_idle_vehicles"]
        zone_open = snapshot_before["zone_open_requests"]
        for vehicle_id, target_zone in decision.rebalances.items():
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle is None or vehicle.status != "idle" or vehicle_id in assigned_vehicles:
                continue
            origin_zone = vehicle.zone
            feasible_destinations = set(
                snapshot_before.get("feasible_destinations", {}).get(
                    origin_zone,
                    self.network.neighbors(origin_zone, radius=1),
                )
            )
            invalid_action = (
                target_zone < 0
                or target_zone >= self.network.zone_count
                or target_zone not in feasible_destinations
            )
            if target_zone < 0 or target_zone >= self.network.zone_count:
                target_zone = vehicle.zone
            if target_zone not in feasible_destinations:
                target_zone = vehicle.zone
            travel_time = 0
            if target_zone != vehicle.zone:
                travel_time = self._actual_travel_time(vehicle.zone, target_zone)
                vehicle.status = "rebalancing"
                vehicle.target_zone = target_zone
                vehicle.remaining_time = travel_time
                vehicle.pickup_remaining_time = 0
                self.metrics["rebalance_time_total"] += travel_time

            reward_result = compute_rebalance_reward(
                config=self.config,
                network=self.network,
                origin_zone=origin_zone,
                target_zone=target_zone,
                travel_time=travel_time,
                zone_open=zone_open,
                zone_idle=zone_idle,
                expected=expected,
                idle_stay_steps=vehicle.idle_stay_steps,
                invalid_action=invalid_action,
            )
            rebalance_reward = reward_result.reward
            vehicle_move_cost = float(reward_result.diagnostics["vehicle_move_cost"])
            self.metrics["vehicle_move_cost_total"] += vehicle_move_cost
            self.metrics["operating_cost_total"] += vehicle_move_cost
            vehicle_rewards[vehicle_id] = vehicle_rewards.get(vehicle_id, 0.0) + rebalance_reward
            rebalance_net_reward = -vehicle_move_cost
            vehicle_net_rewards[vehicle_id] = (
                vehicle_net_rewards.get(vehicle_id, 0.0)
                + rebalance_net_reward
            )
            reward += rebalance_reward
            net_reward += rebalance_net_reward
            component_log = {
                f"reward_{name}": round(value, 3)
                for name, value in reward_result.components.items()
            }
            self.event_log.append(
                {
                    "time": self.time,
                    "event": "invalid_action" if invalid_action else ("rebalance" if reward_result.is_move else "hold"),
                    "vehicle_id": vehicle_id,
                    "request_id": -1,
                    "origin": vehicle.zone,
                    "destination": target_zone,
                    "pickup_time": 0,
                    "trip_time": travel_time,
                    "pickup_grid_distance": 0,
                    "passenger_grid_distance": 0,
                    "grid_distance": int(reward_result.diagnostics["grid_distance"]),
                    "wait_time": 0,
                    "fare": 0.0,
                    "reward": round(rebalance_reward, 3),
                    "reward_components": ",".join(self.config.reward_components),
                    "movement_cost": round(float(reward_result.diagnostics["movement_cost"]), 3),
                    "vehicle_move_cost": round(vehicle_move_cost, 3),
                    "passenger_trip_revenue": 0.0,
                    "operating_cost": round(vehicle_move_cost, 3),
                    "target_idle_penalty": round(float(reward_result.diagnostics["target_idle_penalty"]), 3),
                    "idle_cluster_penalty": round(float(reward_result.diagnostics["idle_cluster_penalty"]), 3),
                    "stay_streak_penalty": round(float(reward_result.diagnostics["stay_streak_penalty"]), 3),
                    "idle_stay_steps": int(reward_result.diagnostics["idle_stay_steps"]),
                    **component_log,
                }
            )
            vehicle.idle_stay_steps = 0 if reward_result.is_move else reward_result.next_stay_steps

        return StepResult(
            time=self.time,
            reward=reward,
            vehicle_rewards=vehicle_rewards,
            served=served,
            canceled=canceled_before,
            new_requests=new_request_count,
            open_requests=len(self.open_requests),
            net_reward=net_reward,
            vehicle_net_rewards=vehicle_net_rewards,
            canceled_by_zone=dict(self._last_canceled_by_zone),
        )

    def _actual_travel_time(self, origin: int, destination: int) -> int:
        if int(origin) == int(destination):
            return 0
        nominal = self.network.travel_time(origin, destination, self.time)
        multiplier = self.scenario.travel_time_multiplier(
            self.network,
            origin,
            destination,
            self.time,
        )
        return max(1, int(math.ceil(nominal * multiplier)))

    def _graph_edge_weights(
        self,
        feasible_destinations: Dict[int, tuple[int, ...]],
        free_flow_travel_times: Dict[tuple[int, int], int],
    ) -> Dict[tuple[int, int], float]:
        """Build the causal symmetric W_t used by graph-wavelet filtering."""

        weights: Dict[tuple[int, int], float] = {}
        for source, destinations in feasible_destinations.items():
            for destination in destinations:
                if source == destination:
                    continue
                reverse_time = free_flow_travel_times.get(
                    (destination, source),
                    free_flow_travel_times[(source, destination)],
                )
                symmetric_time = 0.5 * (
                    float(free_flow_travel_times[(source, destination)])
                    + float(reverse_time)
                )
                bidirectional_flow = (
                    self._historical_od_counts[source, destination]
                    + self._historical_od_counts[destination, source]
                )
                travel_affinity = 1.0 / max(1.0, symmetric_time)
                flow_affinity = 1.0 + math.log1p(float(bidirectional_flow))
                weights[(source, destination)] = travel_affinity * flow_affinity
        return weights

    def _record_scenario_marker(self) -> None:
        marker = self.scenario.marker_at(self.time)
        if marker is None:
            return
        self.event_log.append(
            {
                "time": self.time,
                "event": marker,
                "scenario": self.scenario.family,
                "onset": self.scenario.onset,
                "recovery": self.scenario.recovery,
                "intensity": self.scenario.intensity,
            }
        )

    def _record_logs(
        self,
        new_requests: int,
        step_reward: float,
        step_served: int,
        step_canceled: int,
    ) -> None:
        counts = self._fleet_counts()
        offline = counts["offline"]
        active_vehicle_steps = self.config.fleet_size - offline
        loaded_vehicle_steps = sum(
            1
            for vehicle in self.vehicles.values()
            if vehicle.status == "serving" and vehicle.pickup_remaining_time <= 0
        )
        empty_vehicle_steps = active_vehicle_steps - loaded_vehicle_steps
        self.metrics["empty_vehicle_steps"] += empty_vehicle_steps
        self.metrics["total_active_vehicle_steps"] += active_vehicle_steps
        self.metrics["offline_vehicle_steps"] += offline
        self.metrics["max_offline_vehicles"] = max(
            self.metrics["max_offline_vehicles"],
            float(offline),
        )
        fleet = max(1.0, float(self.config.fleet_size))
        total_cost = float(self.metrics["operating_cost_total"])
        total_revenue = float(self.metrics["revenue"])
        previous_revenue = (
            float(self.summary_log[-1]["revenue_total"]) if self.summary_log else 0.0
        )
        previous_cost = (
            float(self.summary_log[-1]["operating_cost_total"]) if self.summary_log else 0.0
        )

        self.summary_log.append(
            {
                "time": self.time,
                "idle": counts["idle"],
                "serving": counts["serving"],
                "rebalancing": counts["rebalancing"],
                "offline": offline,
                "available": self.config.fleet_size - offline,
                "step_empty_vehicle_steps": empty_vehicle_steps,
                "step_total_active_vehicle_steps": active_vehicle_steps,
                "empty_vehicle_steps_total": self.metrics["empty_vehicle_steps"],
                "total_active_vehicle_steps": self.metrics["total_active_vehicle_steps"],
                "open_requests": len(self.open_requests),
                "new_requests": new_requests,
                "step_served": step_served,
                "step_canceled": step_canceled,
                "step_revenue": round(total_revenue - previous_revenue, 3),
                "step_operating_cost": round(total_cost - previous_cost, 3),
                "served_total": self.metrics["served"],
                "canceled_total": self.metrics["canceled"],
                "generated_total": self.metrics["generated"],
                "revenue_total": round(self.metrics["revenue"], 3),
                "vehicle_move_cost_total": round(self.metrics["vehicle_move_cost_total"], 3),
                "passenger_trip_revenue_total": round(self.metrics["passenger_trip_revenue_total"], 3),
                "operating_cost_total": round(total_cost, 3),
                "profit_total": round(total_revenue - total_cost, 3),
                "avg_vehicle_revenue": total_revenue / fleet,
                "avg_vehicle_cost": total_cost / fleet,
                "avg_vehicle_profit": (total_revenue - total_cost) / fleet,
                "reward_total": round(self.metrics["reward"], 3),
                "step_reward": round(step_reward, 3),
                "demand_estimator": self.demand_history.name,
                "scenario_phase": self.scenario.phase(self.time),
                "scenario_strength": round(self.scenario.strength(self.time), 6),
            }
        )

        if not self.record_detailed_logs:
            return
        open_by_zone = np.zeros(self.network.zone_count, dtype=int)
        for request in self.open_requests.values():
            open_by_zone[request.origin] += 1
        idle_by_zone = np.zeros(self.network.zone_count, dtype=int)
        for vehicle in self.vehicles.values():
            if vehicle.status == "idle":
                idle_by_zone[vehicle.zone] += 1
        expected = self.demand_history.expected_counts(
            self.time,
            self.config.demand_window,
        )
        for vehicle in self.vehicles.values():
            display_zone = (
                vehicle.zone
                if vehicle.status in {"idle", "offline"}
                else vehicle.target_zone if vehicle.target_zone is not None else vehicle.zone
            )
            x, y = self.network.coord(display_zone)
            self.vehicle_log.append(
                {
                    "time": self.time,
                    "vehicle_id": vehicle.vehicle_id,
                    "zone": display_zone,
                    "x": x,
                    "y": y,
                    "status": vehicle.status,
                    "remaining_time": vehicle.remaining_time,
                    "pickup_remaining_time": vehicle.pickup_remaining_time,
                    "passenger_onboard": (
                        vehicle.status == "serving" and vehicle.pickup_remaining_time <= 0
                    ),
                    "idle_stay_steps": vehicle.idle_stay_steps,
                }
            )

        for zone in range(self.network.zone_count):
            x, y = self.network.coord(zone)
            self.zone_log.append(
                {
                    "time": self.time,
                    "zone": zone,
                    "x": x,
                    "y": y,
                    "open_requests": int(open_by_zone[zone]),
                    "idle_vehicles": int(idle_by_zone[zone]),
                    "expected_demand": round(float(expected[zone]), 3),
                    "demand_supply_gap": round(float(open_by_zone[zone] + expected[zone] - idle_by_zone[zone]), 3),
                }
            )

    def _fleet_counts(self) -> Dict[str, int]:
        counts = {"idle": 0, "serving": 0, "rebalancing": 0, "offline": 0}
        for vehicle in self.vehicles.values():
            counts[vehicle.status] = counts.get(vehicle.status, 0) + 1
        return counts

    def logs_as_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        logs = {
            "summary": self.summary_log,
            "vehicles": self.vehicle_log,
            "zones": self.zone_log,
            "events": self.event_log,
            "scenario": [self.scenario.to_dict()],
        }
        if self._bound_policy is not None and hasattr(
            self._bound_policy,
            "dgls_events",
        ):
            logs["dgls"] = list(self._bound_policy.dgls_events)
        return logs
