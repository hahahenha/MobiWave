from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .entities import DispatchDecision
from .dispatch import nearest_assignments
from .graph_wavelet import GraphWaveletNet
from .torch_utils import select_torch_device


class CausalFeatureBuilder:
    """Construct the causal input in Eq. (3) from dispatch snapshots.

    The builder deliberately refuses to synthesize missing history.  A feature
    tensor becomes available only after two complete windows of the longest
    requested horizon have been observed.  Repeated reads at the current time
    are allowed, while inserting a different state at the same time or a past
    state is rejected.
    """

    RAW_FEATURE_NAMES = (
        "open_requests",
        "observed_new_requests",
        "idle_vehicles",
        "serving_vehicles",
        "rebalancing_vehicles",
        "offline_vehicles",
        "passenger_assignments",
        "travel_time_stay",
        "travel_time_east",
        "travel_time_north_east",
        "travel_time_north_west",
        "travel_time_west",
        "travel_time_south_west",
        "travel_time_south_east",
        "move_cost_stay",
        "move_cost_east",
        "move_cost_north_east",
        "move_cost_north_west",
        "move_cost_west",
        "move_cost_south_west",
        "move_cost_south_east",
        "observed_travel_condition",
        "observed_connectivity",
    )

    def __init__(
        self,
        network: Any,
        horizons: Sequence[int] = (1, 2, 4),
        *,
        use_temporal_summaries: bool = True,
        steps_per_hour: float = 1.0,
        hours_per_day: int = 24,
        days_per_week: int = 7,
    ) -> None:
        checked_horizons = tuple(sorted({int(value) for value in horizons}))
        if not checked_horizons or checked_horizons[0] <= 0:
            raise ValueError("horizons must contain positive integers")
        if steps_per_hour <= 0 or hours_per_day <= 0 or days_per_week <= 0:
            raise ValueError("periodic-code intervals must be positive")
        self.network = network
        self.horizons = checked_horizons
        self.use_temporal_summaries = bool(use_temporal_summaries)
        self.steps_per_hour = float(steps_per_hour)
        self.hours_per_day = int(hours_per_day)
        self.days_per_week = int(days_per_week)
        self.raw_feature_dim = len(self.RAW_FEATURE_NAMES)
        self._times: list[int] = []
        self._states: list[np.ndarray] = []
        self._last_snapshot: Mapping[str, Any] | None = None
        self._neighbor_average = self._build_neighbor_average()
        self._neighbor_signature: tuple[tuple[int, int, float], ...] | None = None
        self.feature_slices = self._make_feature_slices()

    @property
    def output_dim(self) -> int:
        temporal_blocks = 2 * len(self.horizons) if self.use_temporal_summaries else 0
        return self.raw_feature_dim * (1 + temporal_blocks) + 6

    @property
    def minimum_history(self) -> int:
        return 2 * max(self.horizons) if self.use_temporal_summaries else 1

    @property
    def last_time(self) -> int | None:
        return self._times[-1] if self._times else None

    @property
    def ready(self) -> bool:
        if len(self._times) < self.minimum_history:
            return False
        tail = self._times[-self.minimum_history :]
        return all(right == left + 1 for left, right in zip(tail, tail[1:]))

    def reset(self) -> None:
        self._times.clear()
        self._states.clear()
        self._last_snapshot = None
        self._neighbor_average = self._build_neighbor_average()
        self._neighbor_signature = None

    def state_from_snapshot(self, snapshot: Mapping[str, Any]) -> np.ndarray:
        node_count = self.network.zone_count
        open_requests = self._zone_array(
            snapshot,
            (
                "zone_backlog_requests"
                if "zone_backlog_requests" in snapshot
                else "zone_open_requests"
            ),
        )
        observed_new = self._zone_array(
            snapshot,
            "zone_new_requests"
            if "zone_new_requests" in snapshot
            else "expected_demand",
        )
        idle = np.zeros(node_count, dtype=np.float32)
        serving = np.zeros(node_count, dtype=np.float32)
        rebalancing = np.zeros(node_count, dtype=np.float32)
        offline = np.zeros(node_count, dtype=np.float32)
        for vehicle in snapshot["vehicles"].values():
            zone = int(vehicle.zone)
            if vehicle.status == "idle":
                idle[zone] += 1.0
            elif vehicle.status == "serving":
                serving[zone] += 1.0
            elif vehicle.status == "rebalancing":
                rebalancing[zone] += 1.0
            elif vehicle.status == "offline":
                offline[zone] += 1.0
        passenger_assignments = np.asarray(
            snapshot.get(
                "zone_passenger_assignments",
                np.zeros(node_count, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        passenger_assignments = self._as_zone_values(
            passenger_assignments,
            "snapshot['zone_passenger_assignments']",
        )
        travel_times, move_costs, external_context = (
            self._candidate_move_features(snapshot)
        )
        return np.stack(
            [
                open_requests,
                observed_new,
                idle,
                serving,
                rebalancing,
                offline,
                passenger_assignments,
                *[travel_times[:, action] for action in range(7)],
                *[move_costs[:, action] for action in range(7)],
                external_context[:, 0],
                external_context[:, 1],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def _candidate_move_features(
        self,
        snapshot: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the seven causal candidate-edge travel/cost channels in X_t."""

        node_count = self.network.zone_count
        travel_times = np.zeros((node_count, 7), dtype=np.float32)
        move_costs = np.zeros((node_count, 7), dtype=np.float32)
        external_context = np.zeros((node_count, 2), dtype=np.float32)
        feasible = snapshot.get("feasible_destinations")
        edge_travel = snapshot.get("edge_travel_times", {})
        time = int(snapshot["time"])
        move_cost_per_grid = float(
            getattr(snapshot.get("config"), "vehicle_move_cost_per_grid", 0.0)
        )
        for source in range(node_count):
            valid_actions = set(self.network.valid_actions(source))
            static_nonstay = {
                int(self.network.move(source, action))
                for action in valid_actions
                if action != 0
            }
            valid_destinations = (
                {
                    int(destination)
                    for destination in feasible.get(source, ())
                }
                if feasible is not None
                else {
                    int(self.network.move(source, action))
                    for action in valid_actions
                }
            )
            for action in range(7):
                destination = int(self.network.move(source, action))
                if action not in valid_actions or destination not in valid_destinations:
                    continue
                edge = (source, destination)
                if edge in edge_travel:
                    travel_times[source, action] = float(edge_travel[edge])
                elif source != destination:
                    travel_times[source, action] = float(
                        self.network.travel_time(source, destination, time)
                    )
                move_costs[source, action] = (
                    move_cost_per_grid
                    * float(self.network.hex_distance(source, destination))
                )
            observed_nonstay = {
                destination
                for destination in valid_destinations
                if destination != source
            }
            travel_ratios = []
            for destination in observed_nonstay:
                nominal = max(
                    1.0,
                    float(self.network.travel_time(source, destination, time)),
                )
                travel_ratios.append(
                    float(edge_travel.get((source, destination), nominal))
                    / nominal
                )
            external_context[source, 0] = (
                float(np.mean(travel_ratios)) if travel_ratios else 1.0
            )
            external_context[source, 1] = (
                float(len(observed_nonstay)) / max(1, len(static_nonstay))
            )
        return travel_times, move_costs, external_context

    def in_transit_arrivals(self, snapshot: Mapping[str, Any]) -> np.ndarray:
        arrivals = np.zeros(self.network.zone_count, dtype=np.float32)
        for vehicle in snapshot["vehicles"].values():
            if (
                vehicle.status in {"serving", "rebalancing"}
                and vehicle.target_zone is not None
            ):
                arrivals[int(vehicle.target_zone)] += 1.0
        return arrivals

    def append_state(
        self,
        time: int,
        state: np.ndarray | torch.Tensor,
        *,
        snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        time = int(time)
        values = self._as_state_array(state)
        if self._times:
            if time < self._times[-1]:
                raise ValueError("causal history cannot accept a past state")
            if time == self._times[-1]:
                if not np.array_equal(values, self._states[-1]):
                    raise ValueError("a different state is already recorded at this time")
                if snapshot is not None:
                    self._last_snapshot = snapshot
                return
        self._times.append(time)
        self._states.append(values.copy())
        self._last_snapshot = snapshot
        # Only the longest two-window history is needed for online features.
        retain = self.minimum_history
        if len(self._times) > retain:
            self._times = self._times[-retain:]
            self._states = self._states[-retain:]

    def observe_snapshot(self, snapshot: Mapping[str, Any]) -> np.ndarray:
        time = int(snapshot["time"])
        state = self.state_from_snapshot(snapshot)
        self.append_state(time, state, snapshot=snapshot)
        return state

    def build(
        self,
        snapshot: Mapping[str, Any] | None = None,
        *,
        remaining_idle: np.ndarray | torch.Tensor | None = None,
        remaining_backlog: np.ndarray | torch.Tensor | None = None,
        remaining_demand: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        if snapshot is not None:
            self.observe_snapshot(snapshot)
        if not self.ready:
            raise RuntimeError(
                f"causal features require {self.minimum_history} consecutive states"
            )
        if self._last_snapshot is None:
            raise RuntimeError("a snapshot is required for current exogenous features")
        time = int(self._last_snapshot["time"])
        if time != self._times[-1]:
            raise RuntimeError("features may only be built for the latest observed state")

        history = np.stack(self._states, axis=0)
        blocks = [history[-1]]
        if self.use_temporal_summaries:
            for horizon in self.horizons:
                current = history[-horizon:].mean(axis=0)
                previous = history[-2 * horizon : -horizon].mean(axis=0)
                blocks.extend((current, current - previous))

        periodic = self.periodic_code(time)
        periodic_nodes = np.broadcast_to(
            periodic[None, :],
            (self.network.zone_count, periodic.shape[0]),
        )
        backlog = (
            self._zone_array(
                self._last_snapshot,
                (
                    "zone_backlog_requests"
                    if "zone_backlog_requests" in self._last_snapshot
                    else "zone_open_requests"
                ),
            )
            if remaining_backlog is None
            else self._as_zone_values(remaining_backlog, "remaining_backlog")
        )
        demand = (
            self._zone_array(
                self._last_snapshot,
                (
                    "zone_new_requests"
                    if "zone_new_requests" in self._last_snapshot
                    else "expected_demand"
                ),
            )
            if remaining_demand is None
            else self._as_zone_values(remaining_demand, "remaining_demand")
        )
        idle = (
            self._zone_array(self._last_snapshot, "zone_idle_vehicles")
            if remaining_idle is None
            else self._as_zone_values(remaining_idle, "remaining_idle")
        )
        feasible_destinations = self._last_snapshot.get("feasible_destinations")
        if feasible_destinations is not None:
            edge_weights = self._last_snapshot.get("edge_graph_weights", {})
            signature = tuple(
                (
                    int(source),
                    int(destination),
                    float(
                        1.0
                        if int(source) == int(destination)
                        else edge_weights.get(
                            (int(source), int(destination)),
                            1.0,
                        )
                    ),
                )
                for source, destinations in sorted(feasible_destinations.items())
                for destination in sorted(destinations)
            )
            if signature != self._neighbor_signature:
                self._neighbor_average = self._build_neighbor_average(
                    feasible_destinations,
                    edge_weights,
                )
                self._neighbor_signature = signature
        neighbor_gap = self._neighbor_average @ (backlog + demand - idle)
        arrivals = self.in_transit_arrivals(self._last_snapshot)
        return np.concatenate(
            [
                *blocks,
                periodic_nodes,
                neighbor_gap[:, None],
                arrivals[:, None],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def periodic_code(self, time: int) -> np.ndarray:
        steps_per_day = self.steps_per_hour * self.hours_per_day
        hour = (float(time) % steps_per_day) / self.steps_per_hour
        weekday = int(math.floor(float(time) / steps_per_day)) % self.days_per_week
        return np.asarray(
            [
                math.sin(2.0 * math.pi * hour / self.hours_per_day),
                math.cos(2.0 * math.pi * hour / self.hours_per_day),
                math.sin(2.0 * math.pi * weekday / self.days_per_week),
                math.cos(2.0 * math.pi * weekday / self.days_per_week),
            ],
            dtype=np.float32,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "horizons": self.horizons,
            "use_temporal_summaries": self.use_temporal_summaries,
            "times": tuple(self._times),
            "states": [state.copy() for state in self._states],
            "last_snapshot": copy.deepcopy(self._last_snapshot),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["horizons"]) != self.horizons:
            raise ValueError("feature-builder horizons do not match")
        if bool(state.get("use_temporal_summaries", True)) != self.use_temporal_summaries:
            raise ValueError("feature-builder temporal-summary mode does not match")
        self.reset()
        for time, values in zip(state["times"], state["states"]):
            self.append_state(int(time), values)
        self._last_snapshot = copy.deepcopy(state.get("last_snapshot"))

    def _make_feature_slices(self) -> dict[str, slice]:
        output: dict[str, slice] = {}
        start = 0
        output["current"] = slice(start, start + self.raw_feature_dim)
        start += self.raw_feature_dim
        if self.use_temporal_summaries:
            for horizon in self.horizons:
                output[f"mean_{horizon}"] = slice(start, start + self.raw_feature_dim)
                start += self.raw_feature_dim
                output[f"delta_{horizon}"] = slice(start, start + self.raw_feature_dim)
                start += self.raw_feature_dim
        output["periodic"] = slice(start, start + 4)
        start += 4
        output["neighbor_gap"] = slice(start, start + 1)
        output["in_transit_arrivals"] = slice(start + 1, start + 2)
        return output

    def _build_neighbor_average(
        self,
        feasible_destinations: Mapping[int, Sequence[int]] | None = None,
        edge_weights: Mapping[tuple[int, int], float] | None = None,
    ) -> np.ndarray:
        node_count = self.network.zone_count
        matrix = np.zeros((node_count, node_count), dtype=np.float32)
        for source in range(node_count):
            destinations = (
                {int(destination) for destination in feasible_destinations[source]}
                if feasible_destinations is not None
                else {
                    int(self.network.move(source, action))
                    for action in self.network.valid_actions(source)
                }
            )
            for destination in destinations:
                matrix[source, destination] = (
                    1.0
                    if destination == source or edge_weights is None
                    else float(edge_weights.get((source, destination), 1.0))
                )
        degrees = matrix.sum(axis=1, keepdims=True)
        return matrix / np.maximum(degrees, 1.0)

    def _as_state_array(self, state: np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        output = np.asarray(state, dtype=np.float32)
        expected = (self.network.zone_count, self.raw_feature_dim)
        if output.shape != expected:
            raise ValueError(f"state must have shape {expected}")
        if not np.isfinite(output).all():
            raise ValueError("state contains non-finite values")
        return output

    def _zone_array(self, snapshot: Mapping[str, Any], key: str) -> np.ndarray:
        output = np.asarray(snapshot[key], dtype=np.float32)
        return self._as_zone_values(output, f"snapshot[{key!r}]")

    def _as_zone_values(
        self,
        values: np.ndarray | torch.Tensor,
        name: str,
    ) -> np.ndarray:
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        output = np.asarray(values, dtype=np.float32)
        expected = (self.network.zone_count,)
        if output.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if not np.isfinite(output).all():
            raise ValueError(f"{name} contains non-finite values")
        return output


class MobiWaveDispatchNet(nn.Module):
    """Graph-wavelet dispatch network with demand, return, and policy heads."""

    STATE_FORMAT_VERSION = 1
    EXTERNAL_CONTEXT_DIM = 2
    _LEGACY_UNUSED_HEAD_KEYS = frozenset(
        {
            "graph_wavelet.out.0.weight",
            "graph_wavelet.out.0.bias",
            "graph_wavelet.out.2.weight",
            "graph_wavelet.out.2.bias",
        }
    )

    def __init__(
        self,
        network: Any,
        input_dim: int,
        hidden_dim: int,
        heat_scales: Sequence[float],
        chebyshev_order: int,
        forecast_horizon: int,
        *,
        service_capacity: float | Sequence[float] = 1.0,
        zone_embedding_dim: int = 0,
        scale_dropout: float = 0.0,
        use_gating: bool = True,
        filter_mode: str = "graph_wavelet",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or forecast_horizon <= 0:
            raise ValueError("model dimensions and forecast horizon must be positive")
        self.network = network
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.forecast_horizon = int(forecast_horizon)
        self.heat_scales = tuple(float(value) for value in heat_scales)
        self.chebyshev_order = int(chebyshev_order)
        service_values = torch.as_tensor(service_capacity, dtype=torch.float32)
        if service_values.ndim == 0:
            service_values = service_values.repeat(network.zone_count)
        if (
            tuple(service_values.shape) != (network.zone_count,)
            or not torch.isfinite(service_values).all()
            or torch.any(service_values <= 0)
        ):
            raise ValueError("service_capacity must be positive for every zone")
        self.register_buffer("service_capacity", service_values)
        self.zone_embedding_dim = int(zone_embedding_dim)
        self.scale_dropout = float(scale_dropout)
        self.use_gating = bool(use_gating)
        self.filter_mode = str(filter_mode)

        self.graph_wavelet = GraphWaveletNet(
            network=network,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heat_scales=self.heat_scales,
            chebyshev_order=chebyshev_order,
            gate_context_dim=self.zone_embedding_dim,
            external_context_dim=self.EXTERNAL_CONTEXT_DIM,
            scale_dropout=self.scale_dropout,
            filter_mode=self.filter_mode,
            prediction_head=False,
        )
        head_hidden = max(8, hidden_dim)
        self.demand_hidden = nn.Linear(hidden_dim, head_hidden)
        self.demand_output = nn.Linear(head_hidden, self.forecast_horizon)
        self.return_hidden = nn.Linear(2 * hidden_dim + 3, head_hidden)
        self.return_output = nn.Linear(head_hidden, 1)
        self.policy_hidden = nn.Linear(2 * hidden_dim + 2, head_hidden)
        self.policy_output = nn.Linear(head_hidden, 1)

        sources, destinations, source_slices = self._build_feasible_edges()
        self.register_buffer("edge_sources", torch.as_tensor(sources, dtype=torch.long))
        self.register_buffer(
            "edge_destinations",
            torch.as_tensor(destinations, dtype=torch.long),
        )
        mask = torch.zeros(
            (network.zone_count, network.zone_count),
            dtype=torch.bool,
        )
        mask[self.edge_sources, self.edge_destinations] = True
        self.register_buffer("feasible_mask", mask)
        self._source_slices = source_slices
        self._topology_signature = tuple(zip(sources, destinations))
        self._graph_weight_signature: tuple[tuple[int, int, float], ...] = ()

    @property
    def edge_count(self) -> int:
        return int(self.edge_sources.numel())

    def forward(
        self,
        x: torch.Tensor,
        *,
        available_supply: torch.Tensor | None = None,
        backlog: torch.Tensor | None = None,
        current_demand: torch.Tensor | None = None,
        dispatch_pressure: torch.Tensor | None = None,
        external_context: torch.Tensor | None = None,
        edge_travel_time: torch.Tensor | None = None,
        edge_move_cost: torch.Tensor | None = None,
        time: int = 0,
        return_probability_matrix: bool = True,
    ) -> dict[str, Any]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, nodes, features]")
        batch_size, node_count, feature_count = x.shape
        if node_count != self.network.zone_count or feature_count != self.input_dim:
            raise ValueError("x has incompatible node or feature dimensions")

        available = self._node_values(
            available_supply,
            batch_size,
            x,
            default=0.0,
            name="available_supply",
        )
        backlog_values = self._node_values(
            backlog,
            batch_size,
            x,
            default=0.0,
            name="backlog",
        )
        demand_values = self._node_values(
            current_demand,
            batch_size,
            x,
            default=0.0,
            name="current_demand",
        )
        if dispatch_pressure is None:
            pressure = (backlog_values + demand_values - available) / (available + 1.0)
        else:
            pressure = self._node_values(
                dispatch_pressure,
                batch_size,
                x,
                default=0.0,
                name="dispatch_pressure",
            )

        _, wavelet_aux = self.graph_wavelet(
            x,
            pressure.unsqueeze(-1),
            external_context=external_context,
            return_aux=True,
        )
        if self.use_gating:
            representation = wavelet_aux["dispatch_representation"]
        else:
            uniform = torch.full_like(
                wavelet_aux["gate_weights"],
                1.0 / wavelet_aux["gate_weights"].shape[-1],
            )
            wavelet_aux["gate_weights"] = uniform
            representation = (
                wavelet_aux["residual"]
                + wavelet_aux["scale_features"].mean(dim=2)
            )
        demand_activation = F.relu(self.demand_hidden(representation))
        demand_prediction = F.softplus(self.demand_output(demand_activation))
        predicted_gap = (
            backlog_values
            + demand_values
            + demand_prediction.sum(dim=-1)
            - self.service_capacity.unsqueeze(0) * available
        )

        source_features = representation[:, self.edge_sources, :]
        destination_features = representation[:, self.edge_destinations, :]
        gap_difference = (
            predicted_gap[:, self.edge_destinations]
            - predicted_gap[:, self.edge_sources]
        )
        travel_default = (
            self._default_edge_travel_time(time)
            if edge_travel_time is None
            else None
        )
        cost_default = self._default_edge_cost() if edge_move_cost is None else None
        travel = self._edge_values(
            edge_travel_time,
            batch_size,
            x,
            name="edge_travel_time",
            default=travel_default,
        )
        cost = self._edge_values(
            edge_move_cost,
            batch_size,
            x,
            name="edge_move_cost",
            default=cost_default,
        )
        return_inputs = torch.cat(
            [
                source_features,
                destination_features,
                gap_difference.unsqueeze(-1),
                travel.unsqueeze(-1),
                cost.unsqueeze(-1),
            ],
            dim=-1,
        )
        return_activation = F.relu(self.return_hidden(return_inputs))
        edge_returns = self.return_output(return_activation).squeeze(-1)
        policy_inputs = torch.cat(
            [
                source_features,
                destination_features,
                gap_difference.unsqueeze(-1),
                edge_returns.unsqueeze(-1),
            ],
            dim=-1,
        )
        policy_activation = F.relu(self.policy_hidden(policy_inputs))
        policy_logits = self.policy_output(policy_activation).squeeze(-1)
        edge_probabilities = self.masked_softmax(policy_logits)
        probability_matrix = None
        if return_probability_matrix:
            probability_matrix = x.new_zeros(
                (batch_size, node_count, node_count)
            )
            probability_matrix[
                :,
                self.edge_sources,
                self.edge_destinations,
            ] = edge_probabilities

        return {
            "demand": demand_prediction,
            "predicted_gap": predicted_gap,
            "edge_returns": edge_returns,
            "policy_logits": policy_logits,
            "edge_probabilities": edge_probabilities,
            "probability_matrix": probability_matrix,
            "edge_sources": self.edge_sources,
            "edge_destinations": self.edge_destinations,
            "feasible_mask": self.feasible_mask,
            "scale_features": wavelet_aux["scale_features"],
            "gate_weights": wavelet_aux["gate_weights"],
            "scale_mask": wavelet_aux["scale_mask"],
            "dispatch_representation": representation,
            "activations": {
                "input_projection": wavelet_aux["input_projection"],
                **{
                    f"scale_encoder_{band}": wavelet_aux["scale_features"][
                        :, :, band, :
                    ]
                    for band in range(wavelet_aux["scale_features"].shape[2])
                },
                **{
                    f"gate_{band}": wavelet_aux["gate_logits"][
                        :, :, band : band + 1
                    ]
                    for band in range(wavelet_aux["gate_weights"].shape[2])
                },
                **(
                    {"zone_embedding": wavelet_aux["zone_embedding"]}
                    if wavelet_aux["zone_embedding"] is not None
                    else {}
                ),
                "residual": wavelet_aux["residual"],
                "dispatch_representation": representation,
                "demand_head": demand_activation,
                "return_head": return_activation,
                "policy_head": policy_activation,
            },
        }

    def update_graph(
        self,
        feasible_destinations: Mapping[int, Sequence[int]] | None,
        edge_weights: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        """Use the observed graph version for filtering and action masking."""

        if feasible_destinations is None:
            normalized = None
        else:
            normalized = {
                int(source): tuple(sorted({int(value) for value in destinations}))
                for source, destinations in feasible_destinations.items()
            }
            if set(normalized) != set(range(self.network.zone_count)):
                raise ValueError("feasible_destinations must define every source zone")
            for source, destinations in normalized.items():
                if source not in destinations:
                    raise ValueError("every source must retain its stay action")
                if any(
                    destination < 0 or destination >= self.network.zone_count
                    for destination in destinations
                ):
                    raise ValueError("feasible destination is outside the road graph")
        sources, destinations, source_slices = self._build_feasible_edges(normalized)
        topology_signature = tuple(zip(sources, destinations))
        graph_weight_signature = tuple(
            (
                int(source),
                int(destination),
                0.0
                if int(source) == int(destination)
                else float(
                    1.0
                    if edge_weights is None
                    else edge_weights.get((int(source), int(destination)), 1.0)
                ),
            )
            for source, destination in topology_signature
        )
        if any(
            not math.isfinite(weight) or (source != destination and weight <= 0.0)
            for source, destination, weight in graph_weight_signature
        ):
            raise ValueError("edge_weights must be finite and positive")
        topology_changed = topology_signature != self._topology_signature
        weights_changed = graph_weight_signature != self._graph_weight_signature
        if topology_changed:
            device = self.edge_sources.device
            self.edge_sources = torch.as_tensor(
                sources,
                dtype=torch.long,
                device=device,
            )
            self.edge_destinations = torch.as_tensor(
                destinations,
                dtype=torch.long,
                device=device,
            )
            mask = torch.zeros(
                (self.network.zone_count, self.network.zone_count),
                dtype=torch.bool,
                device=device,
            )
            mask[self.edge_sources, self.edge_destinations] = True
            self.feasible_mask = mask
            self._source_slices = source_slices
            self._topology_signature = topology_signature
        if topology_changed or weights_changed:
            self.graph_wavelet.update_topology(
                self.network,
                normalized,
                edge_weights,
            )
            self._graph_weight_signature = graph_weight_signature

    def masked_softmax(self, edge_logits: torch.Tensor) -> torch.Tensor:
        if edge_logits.ndim != 2 or edge_logits.shape[1] != self.edge_count:
            raise ValueError("edge_logits must have shape [batch, feasible_edges]")
        batch_size = edge_logits.shape[0]
        source_index = self.edge_sources.unsqueeze(0).expand(batch_size, -1)
        maxima = edge_logits.new_full(
            (batch_size, self.network.zone_count),
            -torch.inf,
        )
        maxima.scatter_reduce_(
            1,
            source_index,
            edge_logits,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(
            edge_logits - maxima.gather(1, source_index)
        )
        normalizers = edge_logits.new_zeros(
            (batch_size, self.network.zone_count)
        )
        normalizers.scatter_add_(1, source_index, exponentials)
        return exponentials / normalizers.gather(
            1,
            source_index,
        ).clamp_min(torch.finfo(edge_logits.dtype).tiny)

    @torch.no_grad()
    def allocate(
        self,
        edge_probabilities: torch.Tensor,
        available_supply: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if edge_probabilities.ndim == 1:
            edge_probabilities = edge_probabilities.unsqueeze(0)
        if edge_probabilities.ndim != 2 or edge_probabilities.shape[1] != self.edge_count:
            raise ValueError("edge_probabilities must have one value per feasible edge")
        batch_size = edge_probabilities.shape[0]
        available = torch.as_tensor(
            available_supply,
            device=edge_probabilities.device,
        )
        if available.ndim == 1:
            available = available.unsqueeze(0)
        if available.shape != (batch_size, self.network.zone_count):
            raise ValueError("available_supply must have shape [batch, nodes]")
        if not torch.isfinite(available.float()).all():
            raise ValueError("available_supply contains non-finite values")
        rounded = available.round()
        if (available.float() - rounded.float()).abs().max().item() > 1e-6:
            raise ValueError("available_supply must contain integers")
        if (rounded < 0).any():
            raise ValueError("available_supply must be nonnegative")
        available = rounded.to(dtype=torch.long)
        stochastic = self.training if stochastic is None else bool(stochastic)
        allocation = torch.zeros(
            (batch_size, self.edge_count),
            dtype=torch.long,
            device=edge_probabilities.device,
        )
        available_counts = available.detach().cpu().tolist()
        edge_destinations = (
            self.edge_destinations.detach().cpu().tolist()
            if not stochastic
            else ()
        )

        for batch in range(batch_size):
            for source, (start, stop) in enumerate(self._source_slices):
                count = int(available_counts[batch][source])
                if count == 0:
                    continue
                probabilities = edge_probabilities[batch, start:stop]
                probabilities = probabilities / probabilities.sum().clamp_min(
                    torch.finfo(probabilities.dtype).eps
                )
                if stochastic:
                    samples = torch.multinomial(
                        probabilities,
                        count,
                        replacement=True,
                        generator=generator,
                    )
                    allocation[batch, start:stop] = torch.bincount(
                        samples,
                        minlength=stop - start,
                    )
                else:
                    expected = probabilities * count
                    base = torch.floor(expected).to(dtype=torch.long)
                    remaining = count - int(base.sum().item())
                    allocation[batch, start:stop] = base
                    if remaining:
                        fractions = (expected - base).detach().cpu().tolist()
                        destinations = edge_destinations[start:stop]
                        order = sorted(
                            range(stop - start),
                            key=lambda index: (-fractions[index], destinations[index]),
                        )
                        for index in order[:remaining]:
                            allocation[batch, start + index] += 1

        source_index = self.edge_sources.unsqueeze(0).expand(batch_size, -1)
        allocated_by_source = allocation.new_zeros(
            (batch_size, self.network.zone_count)
        )
        allocated_by_source.scatter_add_(
            1,
            source_index,
            allocation,
        )
        if not torch.equal(allocated_by_source, available):
            raise RuntimeError("integer allocation failed fleet conservation")
        return allocation

    def dense_flow(self, edge_allocation: torch.Tensor) -> torch.Tensor:
        if edge_allocation.ndim == 1:
            edge_allocation = edge_allocation.unsqueeze(0)
        if edge_allocation.ndim != 2 or edge_allocation.shape[1] != self.edge_count:
            raise ValueError("edge_allocation must have one value per feasible edge")
        output = edge_allocation.new_zeros(
            (
                edge_allocation.shape[0],
                self.network.zone_count,
                self.network.zone_count,
            )
        )
        output[:, self.edge_sources, self.edge_destinations] = edge_allocation
        return output

    def flow_log_probability(
        self,
        edge_probabilities: torch.Tensor,
        edge_allocation: torch.Tensor,
    ) -> torch.Tensor:
        if edge_probabilities.shape != edge_allocation.shape:
            raise ValueError("probabilities and allocations must have the same shape")
        batch_size = edge_probabilities.shape[0]
        tiny = torch.finfo(edge_probabilities.dtype).tiny
        counts = edge_allocation.to(edge_probabilities.dtype)
        source_index = self.edge_sources.unsqueeze(0).expand(batch_size, -1)
        totals = edge_probabilities.new_zeros(
            (batch_size, self.network.zone_count)
        )
        totals.scatter_add_(1, source_index, counts)
        coefficient = (
            torch.lgamma(totals + 1.0).sum(dim=-1)
            - torch.lgamma(counts + 1.0).sum(dim=-1)
        )
        likelihood = (
            counts * edge_probabilities.clamp_min(tiny).log()
        ).sum(dim=-1)
        return coefficient + likelihood

    def joint_loss(
        self,
        outputs: Mapping[str, Any],
        *,
        demand_targets: torch.Tensor | None = None,
        return_targets: torch.Tensor | None = None,
        return_mask: torch.Tensor | None = None,
        action_allocation: torch.Tensor | None = None,
        old_log_probabilities: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        demand_weight: float = 1.0,
        return_weight: float = 1.0,
        policy_weight: float = 1.0,
        gate_balance_weight: float = 0.01,
        demand_huber_delta: float = 1.0,
        return_huber_delta: float = 1.0,
        policy_clip: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        reference = outputs["demand"]
        zero = reference.sum() * 0.0
        demand_loss = zero
        if demand_targets is not None:
            targets = demand_targets.to(device=reference.device, dtype=reference.dtype)
            if targets.shape != reference.shape:
                raise ValueError("demand_targets must match the H-step demand prediction")
            demand_loss = F.huber_loss(
                reference,
                targets,
                reduction="mean",
                delta=float(demand_huber_delta),
            )

        return_loss = zero
        if return_targets is not None:
            predicted_returns = outputs["edge_returns"]
            targets = return_targets.to(
                device=predicted_returns.device,
                dtype=predicted_returns.dtype,
            )
            if targets.shape != predicted_returns.shape:
                raise ValueError("return_targets must match edge_returns")
            elementwise = F.huber_loss(
                predicted_returns,
                targets,
                reduction="none",
                delta=float(return_huber_delta),
            )
            if return_mask is not None:
                mask = return_mask.to(device=elementwise.device, dtype=torch.bool)
                if mask.shape != elementwise.shape:
                    raise ValueError("return_mask must match edge_returns")
                return_loss = elementwise[mask].mean() if mask.any() else zero
            else:
                return_loss = elementwise.mean()

        supplied_policy_terms = (
            action_allocation,
            old_log_probabilities,
            advantages,
        )
        policy_loss = zero
        if any(term is not None for term in supplied_policy_terms):
            if not all(term is not None for term in supplied_policy_terms):
                raise ValueError(
                    "action_allocation, old_log_probabilities, and advantages are required together"
                )
            current_log_probability = self.flow_log_probability(
                outputs["edge_probabilities"],
                action_allocation.to(outputs["edge_probabilities"].device),
            )
            old = old_log_probabilities.to(
                device=current_log_probability.device,
                dtype=current_log_probability.dtype,
            ).reshape(-1)
            advantage = advantages.to(
                device=current_log_probability.device,
                dtype=current_log_probability.dtype,
            ).reshape(-1)
            if old.shape != current_log_probability.shape or advantage.shape != old.shape:
                raise ValueError("old log probabilities and advantages need one value per batch")
            ratio = torch.exp((current_log_probability - old).clamp(-20.0, 20.0))
            clipped = ratio.clamp(1.0 - float(policy_clip), 1.0 + float(policy_clip))
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()

        mean_gate = outputs["gate_weights"].mean(dim=(0, 1))
        gate_balance_loss = ((mean_gate - 1.0 / mean_gate.numel()) ** 2).sum()
        total = (
            float(demand_weight) * demand_loss
            + float(return_weight) * return_loss
            + float(policy_weight) * policy_loss
            + float(gate_balance_weight) * gate_balance_loss
        )
        return {
            "total": total,
            "demand": demand_loss,
            "return": return_loss,
            "policy": policy_loss,
            "gate_balance": gate_balance_loss,
        }

    def export_state(
        self,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = {
            "format_version": self.STATE_FORMAT_VERSION,
            "metadata": {
                "node_count": self.network.zone_count,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "forecast_horizon": self.forecast_horizon,
                "heat_scales": self.heat_scales,
                "chebyshev_order": self.chebyshev_order,
                "service_capacity": self.service_capacity.detach().cpu(),
                "zone_embedding_dim": self.zone_embedding_dim,
                "scale_dropout": self.scale_dropout,
                "use_gating": self.use_gating,
                "filter_mode": self.filter_mode,
                "edge_sources": self.edge_sources.detach().cpu(),
                "edge_destinations": self.edge_destinations.detach().cpu(),
            },
            "state_dict": copy.deepcopy(self.state_dict()),
        }
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
        return payload

    def load_exported_state(
        self,
        source: Mapping[str, Any] | str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> None:
        if isinstance(source, (str, Path)):
            try:
                payload = torch.load(
                    source,
                    map_location=map_location,
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(source, map_location=map_location)
        else:
            payload = source
        if int(payload.get("format_version", -1)) != self.STATE_FORMAT_VERSION:
            raise ValueError("unsupported MobiWave state format")
        metadata = payload["metadata"]
        expected = {
            "node_count": self.network.zone_count,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "forecast_horizon": self.forecast_horizon,
            "heat_scales": self.heat_scales,
            "chebyshev_order": self.chebyshev_order,
            "service_capacity": self.service_capacity.detach().cpu(),
            "zone_embedding_dim": self.zone_embedding_dim,
            "scale_dropout": self.scale_dropout,
            "use_gating": self.use_gating,
            "filter_mode": self.filter_mode,
        }
        for key, expected_value in expected.items():
            actual = metadata[key]
            if key == "heat_scales":
                actual = tuple(actual)
            if key == "service_capacity":
                if not torch.equal(
                    torch.as_tensor(actual),
                    torch.as_tensor(expected_value),
                ):
                    raise ValueError(f"state metadata mismatch for {key}")
                continue
            if actual != expected_value:
                raise ValueError(f"state metadata mismatch for {key}")
        if not torch.equal(
            torch.as_tensor(metadata["edge_sources"]),
            self.edge_sources.detach().cpu(),
        ) or not torch.equal(
            torch.as_tensor(metadata["edge_destinations"]),
            self.edge_destinations.detach().cpu(),
        ):
            raise ValueError("state feasible-edge ordering does not match")
        self.load_state_dict(
            self._migrate_legacy_parameter_state(payload["state_dict"])
        )

    def _migrate_legacy_parameter_state(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Remove only the complete, formerly unused graph prediction head."""

        present = self._LEGACY_UNUSED_HEAD_KEYS.intersection(state)
        if not present:
            return state
        if present != self._LEGACY_UNUSED_HEAD_KEYS:
            raise ValueError(
                "legacy graph-wavelet prediction head is incomplete"
            )
        expected_shapes = {
            "graph_wavelet.out.0.weight": (
                self.hidden_dim,
                self.hidden_dim,
            ),
            "graph_wavelet.out.0.bias": (self.hidden_dim,),
            "graph_wavelet.out.2.weight": (1, self.hidden_dim),
            "graph_wavelet.out.2.bias": (1,),
        }
        for name, shape in expected_shapes.items():
            try:
                value = torch.as_tensor(state[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid legacy graph-wavelet parameter {name}"
                ) from exc
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(
                    f"invalid legacy graph-wavelet parameter {name}"
                )
        return {
            name: value
            for name, value in state.items()
            if name not in self._LEGACY_UNUSED_HEAD_KEYS
        }

    def _build_feasible_edges(
        self,
        feasible_destinations: Mapping[int, Sequence[int]] | None = None,
    ) -> tuple[list[int], list[int], tuple[tuple[int, int], ...]]:
        sources: list[int] = []
        destinations: list[int] = []
        source_slices: list[tuple[int, int]] = []
        for source in range(self.network.zone_count):
            start = len(sources)
            valid_destinations = (
                sorted({int(value) for value in feasible_destinations[source]})
                if feasible_destinations is not None
                else sorted(
                    {
                        int(self.network.move(source, action))
                        for action in self.network.valid_actions(source)
                    }
                )
            )
            for destination in valid_destinations:
                sources.append(source)
                destinations.append(destination)
            source_slices.append((start, len(sources)))
        return sources, destinations, tuple(source_slices)

    def _node_values(
        self,
        values: torch.Tensor | None,
        batch_size: int,
        reference: torch.Tensor,
        *,
        default: float,
        name: str,
    ) -> torch.Tensor:
        if values is None:
            return reference.new_full(
                (batch_size, self.network.zone_count),
                float(default),
            )
        output = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.shape == (1, self.network.zone_count) and batch_size > 1:
            output = output.expand(batch_size, -1)
        if output.shape != (batch_size, self.network.zone_count):
            raise ValueError(f"{name} must have shape [batch, nodes]")
        return output

    def _edge_values(
        self,
        values: torch.Tensor | None,
        batch_size: int,
        reference: torch.Tensor,
        *,
        name: str,
        default: np.ndarray | None,
    ) -> torch.Tensor:
        source = default if values is None else values
        if source is None:
            raise RuntimeError(f"{name} has neither an explicit value nor a default")
        output = torch.as_tensor(source, device=reference.device, dtype=reference.dtype)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.shape == (1, self.edge_count) and batch_size > 1:
            output = output.expand(batch_size, -1)
        if output.shape != (batch_size, self.edge_count):
            raise ValueError(f"{name} must have shape [batch, feasible_edges]")
        return output

    def _default_edge_travel_time(self, time: int) -> np.ndarray:
        return np.asarray(
            [
                self.network.travel_time(int(source), int(destination), int(time))
                for source, destination in zip(
                    self.edge_sources.detach().cpu().tolist(),
                    self.edge_destinations.detach().cpu().tolist(),
                )
            ],
            dtype=np.float32,
        )

    def _default_edge_cost(self) -> np.ndarray:
        return np.asarray(
            [
                self.network.hex_distance(int(source), int(destination))
                for source, destination in zip(
                    self.edge_sources.detach().cpu().tolist(),
                    self.edge_destinations.detach().cpu().tolist(),
                )
            ],
            dtype=np.float32,
        )


class MobiWavePolicy:
    """Executable dispatch policy for the paper's graph-wavelet module."""

    name = "MobiWave"

    def __init__(
        self,
        config: Any,
        network: Any,
        *,
        horizons: Sequence[int] | None = None,
        forecast_horizon: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.network = network
        configured_horizons = tuple(
            horizons
            if horizons is not None
            else getattr(config, "mobiwave_temporal_horizons", (1, 2, 4))
        )
        self.feature_builder = CausalFeatureBuilder(
            network,
            configured_horizons,
            use_temporal_summaries=bool(
                getattr(config, "mobiwave_use_temporal_summaries", True)
            ),
            steps_per_hour=float(getattr(config, "mobiwave_steps_per_hour", 1.0)),
        )
        self.device = torch.device(device or select_torch_device())
        self.model = MobiWaveDispatchNet(
            network=network,
            input_dim=self.feature_builder.output_dim,
            hidden_dim=int(
                config.mobiwave_hidden_dim
            ),
            heat_scales=tuple(config.graph_wavelet_heat_scales),
            chebyshev_order=int(config.graph_wavelet_chebyshev_order),
            forecast_horizon=int(
                forecast_horizon
                or getattr(config, "mobiwave_forecast_horizon", config.demand_window)
            ),
            zone_embedding_dim=int(
                getattr(config, "mobiwave_zone_embedding_dim", 0)
            ),
            scale_dropout=float(getattr(config, "mobiwave_scale_dropout", 0.0)),
            use_gating=bool(getattr(config, "mobiwave_use_gating", True)),
            filter_mode=str(
                getattr(config, "mobiwave_filter_mode", "graph_wavelet")
            ),
            service_capacity=float(
                getattr(config, "mobiwave_service_capacity", 1.0)
            ),
        ).to(self.device)
        self.model.eval()
        self.last_outputs: dict[str, Any] | None = None
        self.last_edge_allocation: torch.Tensor | None = None
        self._last_flow_matrix_cache: torch.Tensor | None = None
        self.last_vehicle_edges: dict[int, tuple[int, int]] = {}
        self.last_features: np.ndarray | None = None
        self.last_available_supply: np.ndarray | None = None
        self.last_backlog: np.ndarray | None = None
        self.last_current_demand: np.ndarray | None = None
        self.last_external_context: np.ndarray | None = None
        self.current_graph_version = ""
        # ``None`` preserves the standalone model.train()/eval() convention.
        # DGLS sets this explicitly: stochastic only for the offline behavior
        # rollout and deterministic throughout the online test stream.
        self.stochastic_dispatch: bool | None = None

    def reset(self) -> None:
        self.feature_builder.reset()
        self.last_outputs = None
        self.last_edge_allocation = None
        self._last_flow_matrix_cache = None
        self.last_vehicle_edges = {}
        self.last_features = None
        self.last_available_supply = None
        self.last_backlog = None
        self.last_current_demand = None
        self.last_external_context = None
        self.current_graph_version = ""

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()

    @property
    def last_flow_matrix(self) -> torch.Tensor | None:
        """Materialize the dense flow only for diagnostics that request it."""

        if (
            self._last_flow_matrix_cache is None
            and self.last_edge_allocation is not None
        ):
            self._last_flow_matrix_cache = self.model.dense_flow(
                self.last_edge_allocation
            )
        return self._last_flow_matrix_cache

    def observe_warmup_snapshot(
        self,
        snapshot: Mapping[str, Any],
        assignments: Mapping[int, int],
    ) -> None:
        """Accumulate X_t before t_0 without executing a model dispatch."""

        time = int(snapshot["time"])
        if (
            time == 0
            and self.feature_builder.last_time is not None
            and self.feature_builder.last_time > 0
        ):
            self.reset()
        self.model.update_graph(
            snapshot.get("feasible_destinations"),
            snapshot.get("edge_graph_weights"),
        )
        self.current_graph_version = str(snapshot.get("graph_version", ""))
        self.feature_builder.observe_snapshot(
            self._snapshot_with_assignments(snapshot, assignments)
        )
        self.last_outputs = None
        self.last_edge_allocation = None
        self._last_flow_matrix_cache = None
        self.last_vehicle_edges = {}
        self.last_features = None
        self.last_available_supply = None
        self.last_backlog = None
        self.last_current_demand = None
        self.last_external_context = None

    def _snapshot_with_assignments(
        self,
        snapshot: Mapping[str, Any],
        assignments: Mapping[int, int],
    ) -> dict[str, Any]:
        assignment_counts = np.zeros(
            self.network.zone_count,
            dtype=np.float32,
        )
        for request_id in assignments.values():
            request = snapshot["requests"].get(int(request_id))
            if request is not None:
                assignment_counts[int(request.origin)] += 1.0
        feature_snapshot = dict(snapshot)
        feature_snapshot["zone_passenger_assignments"] = assignment_counts
        return feature_snapshot

    def decide(self, snapshot: Mapping[str, Any]) -> DispatchDecision:
        time = int(snapshot["time"])
        if (
            time == 0
            and self.feature_builder.last_time is not None
            and self.feature_builder.last_time > 0
        ):
            self.reset()
        assignments = nearest_assignments(dict(snapshot))
        feature_snapshot = self._snapshot_with_assignments(
            snapshot,
            assignments,
        )
        self.model.update_graph(
            snapshot.get("feasible_destinations"),
            snapshot.get("edge_graph_weights"),
        )
        self.current_graph_version = str(snapshot.get("graph_version", ""))
        self.feature_builder.observe_snapshot(feature_snapshot)
        if not self.feature_builder.ready:
            self.last_outputs = None
            self.last_edge_allocation = None
            self._last_flow_matrix_cache = None
            self.last_vehicle_edges = {}
            self.last_features = None
            self.last_available_supply = None
            self.last_backlog = None
            self.last_current_demand = None
            self.last_external_context = None
            return DispatchDecision(
                assignments=assignments,
                rebalances={},
            )

        remaining_by_zone: dict[int, list[int]] = {
            zone: [] for zone in range(self.network.zone_count)
        }
        assigned_vehicle_ids = set(assignments)
        for vehicle_id, vehicle in snapshot["vehicles"].items():
            if vehicle.status == "idle" and vehicle_id not in assigned_vehicle_ids:
                remaining_by_zone[int(vehicle.zone)].append(int(vehicle_id))
        for vehicle_ids in remaining_by_zone.values():
            vehicle_ids.sort()
        available = np.asarray(
            [len(remaining_by_zone[zone]) for zone in range(self.network.zone_count)],
            dtype=np.int64,
        )
        backlog = np.asarray(
            snapshot.get(
                "zone_backlog_requests",
                snapshot["zone_open_requests"],
            ),
            dtype=np.float32,
        ).copy()
        demand = np.asarray(
            snapshot.get("zone_new_requests", snapshot["expected_demand"]),
            dtype=np.float32,
        ).copy()
        current_request_ids = {
            int(request.request_id)
            for request in snapshot.get("new_requests", ())
        }
        for request_id in assignments.values():
            request = snapshot["requests"].get(request_id)
            if request is None:
                continue
            target = demand if int(request_id) in current_request_ids else backlog
            target[int(request.origin)] = max(
                0.0,
                float(target[int(request.origin)]) - 1.0,
            )
        pressure = (backlog + demand - available) / (available + 1.0)
        features = self.feature_builder.build(
            remaining_idle=available,
            remaining_backlog=backlog,
            remaining_demand=demand,
        )
        external_indices = [
            self.feature_builder.RAW_FEATURE_NAMES.index(
                "observed_travel_condition"
            ),
            self.feature_builder.RAW_FEATURE_NAMES.index(
                "observed_connectivity"
            ),
        ]
        external_context = features[:, external_indices].copy()

        x = torch.as_tensor(features, device=self.device).unsqueeze(0)
        available_tensor = torch.as_tensor(available, device=self.device).unsqueeze(0)
        edge_travel_lookup = snapshot.get("edge_travel_times", {})
        edge_travel = torch.as_tensor(
            [
                float(
                    edge_travel_lookup[
                        (int(source), int(destination))
                    ]
                    if (int(source), int(destination)) in edge_travel_lookup
                    else self.network.travel_time(
                            int(source),
                            int(destination),
                            time,
                        )
                )
                for source, destination in zip(
                    self.model.edge_sources.detach().cpu().tolist(),
                    self.model.edge_destinations.detach().cpu().tolist(),
                )
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        edge_cost = torch.as_tensor(
            [
                float(self.config.vehicle_move_cost_per_grid)
                * self.network.hex_distance(int(source), int(destination))
                for source, destination in zip(
                    self.model.edge_sources.detach().cpu().tolist(),
                    self.model.edge_destinations.detach().cpu().tolist(),
                )
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(
                x,
                available_supply=available_tensor,
                backlog=torch.as_tensor(backlog, device=self.device).unsqueeze(0),
                current_demand=torch.as_tensor(demand, device=self.device).unsqueeze(0),
                dispatch_pressure=torch.as_tensor(pressure, device=self.device).unsqueeze(0),
                external_context=torch.as_tensor(
                    external_context,
                    device=self.device,
                ).unsqueeze(0),
                edge_travel_time=edge_travel,
                edge_move_cost=edge_cost,
                time=time,
                return_probability_matrix=False,
            )
            edge_allocation = self.model.allocate(
                outputs["edge_probabilities"],
                available_tensor,
                stochastic=(
                    self.model.training
                    if self.stochastic_dispatch is None
                    else self.stochastic_dispatch
                ),
            )

        rebalances: dict[int, int] = {}
        vehicle_edges: dict[int, tuple[int, int]] = {}
        edge_sources = self.model.edge_sources.detach().cpu().tolist()
        edge_destinations = self.model.edge_destinations.detach().cpu().tolist()
        edge_counts = edge_allocation[0].detach().cpu().tolist()
        cursor_by_source = {zone: 0 for zone in range(self.network.zone_count)}
        for source, destination, count in zip(
            edge_sources,
            edge_destinations,
            edge_counts,
        ):
            cursor = cursor_by_source[source]
            selected = remaining_by_zone[source][cursor : cursor + int(count)]
            cursor_by_source[source] += int(count)
            for vehicle_id in selected:
                rebalances[vehicle_id] = int(destination)
                vehicle_edges[vehicle_id] = (int(source), int(destination))
        if any(
            cursor_by_source[zone] != len(remaining_by_zone[zone])
            for zone in remaining_by_zone
        ):
            raise RuntimeError("vehicle-to-flow conversion violated fleet conservation")

        self.last_outputs = outputs
        self.last_edge_allocation = edge_allocation.detach().cpu()
        self._last_flow_matrix_cache = None
        self.last_vehicle_edges = vehicle_edges
        self.last_features = features.copy()
        self.last_available_supply = available.copy()
        self.last_backlog = backlog.copy()
        self.last_current_demand = demand.copy()
        self.last_external_context = external_context.copy()
        return DispatchDecision(
            assignments=assignments,
            rebalances=rebalances,
        )

    def dgls_evidence(self) -> dict[str, Any]:
        if self.last_outputs is None:
            raise RuntimeError("no MobiWave forward pass has been recorded")
        return {
            "scale_features": self.last_outputs["scale_features"].detach().cpu(),
            "gate_weights": self.last_outputs["gate_weights"].detach().cpu(),
            "scale_mask": self.last_outputs["scale_mask"].detach().cpu(),
            "dispatch_representation": self.last_outputs[
                "dispatch_representation"
            ].detach().cpu(),
            "activations": {
                name: value.detach().cpu()
                for name, value in self.last_outputs["activations"].items()
            },
        }

    def export_state(self, path: str | Path | None = None) -> dict[str, Any]:
        payload = {
            "format_version": 1,
            "model": self.model.export_state(),
            "feature_builder": self.feature_builder.state_dict(),
            "training": bool(self.model.training),
            "stochastic_dispatch": self.stochastic_dispatch,
            "current_graph_version": self.current_graph_version,
        }
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
        return payload

    def load_exported_state(
        self,
        source: Mapping[str, Any] | str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> None:
        if isinstance(source, (str, Path)):
            try:
                payload = torch.load(
                    source,
                    map_location=map_location or self.device,
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(source, map_location=map_location or self.device)
        else:
            payload = source
        if "model" not in payload:
            # Backward-compatible loading of an older model-only artifact.
            self.model.load_exported_state(payload, map_location=map_location or self.device)
            return
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError("unsupported MobiWave policy state format")
        metadata = payload["model"]["metadata"]
        sources = torch.as_tensor(metadata["edge_sources"]).tolist()
        destinations = torch.as_tensor(metadata["edge_destinations"]).tolist()
        feasible = {zone: [] for zone in range(self.network.zone_count)}
        for source_zone, destination_zone in zip(sources, destinations):
            feasible[int(source_zone)].append(int(destination_zone))
        self.model.update_graph(feasible)
        self.model.load_exported_state(
            payload["model"],
            map_location=map_location or self.device,
        )
        self.feature_builder.load_state_dict(payload["feature_builder"])
        self.current_graph_version = str(payload.get("current_graph_version", ""))
        self.model.train(bool(payload.get("training", False)))
        value = payload.get("stochastic_dispatch")
        self.stochastic_dispatch = None if value is None else bool(value)


__all__ = [
    "CausalFeatureBuilder",
    "MobiWaveDispatchNet",
    "MobiWavePolicy",
]
