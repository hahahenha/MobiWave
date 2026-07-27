from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
from contextlib import contextmanager
import math
import time as wall_time

import numpy as np
import torch

from .adaptation.controller import (
    ControllerConfig,
    DGLSTransactionController,
    ValidationPair,
)
from .adaptation.dgls import (
    deterministic_row_sample,
    rbf_mmd2,
)
from .adaptation.replay import (
    EnvCheckpoint,
    ExogenousTape,
    PairedSuffixValidator,
    ReplayOutcome,
)
from .adaptation.state import AcceptedState, CausalSplit, LayerSignals, MutableState
from .environment import DispatchEnv
from .optimizer import DGLSFastSlowOptimizer
from .model import MobiWaveDispatchNet, MobiWavePolicy


REQUIRED_VIOLATIONS = (
    "maximum_wait_breaches",
    "infeasible_moves",
    "zone_service_shortfall",
)


@dataclass
class _PendingTransition:
    record: dict[str, Any]
    vehicle_edges: dict[int, tuple[int, int]]


class DGLSMobiWavePolicy(MobiWavePolicy):
    """End-to-end MobiWave policy following the paper's online algorithm.

    The live model is always the last accepted model. Candidate training occurs
    on an isolated clone, and paired suffix replay is the only path by which a
    candidate can replace the accepted parameters, optimizer memories,
    importance estimate, and reference FIFO.
    """

    name = "MobiWave"

    def __init__(
        self,
        config: Any,
        network: Any,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, network, device=device)
        self._env: DispatchEnv | None = None
        self._records: list[dict[str, Any]] = []
        self._pending: _PendingTransition | None = None
        self._checkpoints: dict[int, EnvCheckpoint] = {}
        self._builder_before: dict[int, dict[str, Any]] = {}
        self._controller: DGLSTransactionController | None = None
        self._optimizer: DGLSFastSlowOptimizer | None = None
        self._importance: dict[str, Any] | None = None
        self._reference_records: list[dict[str, Any]] = []
        self._layer_bandwidths: dict[str, float] = {}
        self._reward_mean = 0.0
        self._reward_count = 0
        self._last_attempt_time = -1
        self._validation_audit: dict[str, Any] = {}
        self._runtime_events: list[dict[str, Any]] = []
        self._offline_collection = False
        self._reactivation_pending = True
        # Simulation time restarts at zero for offline and test streams.  These
        # monotone identities preserve true accepted-record FIFO order across
        # that boundary and prevent equal timestamps from being conflated.
        self._stream_serial = 0
        self._record_serial = 0
        self._active_stream_id = "unbound:0"
        self._historical_od_prior: np.ndarray | None = None
        # Cache complete, detached scale features per accepted reference
        # record.  A record-level cache avoids repeating the same expensive
        # graph-wavelet forward when later drift checks request other zones
        # from that record.  The small LRU bound keeps host memory predictable.
        self._spectral_reference_cache: OrderedDict[
            str,
            torch.Tensor,
        ] = OrderedDict()
        self._reference_match_source: Sequence[Mapping[str, Any]] | None = None
        self._reference_match_source_length = 0
        self._reference_match_metadata_cache: tuple[Any, ...] | None = None
        self._graph_record_cache: OrderedDict[
            tuple[tuple[int, int, float], ...],
            tuple[
                dict[int, tuple[int, ...]],
                dict[tuple[int, int], float],
                dict[tuple[int, int], float],
                tuple[tuple[int, int, float], ...],
            ],
        ] = OrderedDict()

    @property
    def controller(self) -> DGLSTransactionController | None:
        return self._controller

    @property
    def dgls_events(self) -> tuple[dict[str, Any], ...]:
        controller_events = () if self._controller is None else self._controller.events
        return tuple(copy.deepcopy([*controller_events, *self._runtime_events]))

    def bind_environment(self, env: DispatchEnv) -> None:
        if self._controller is None and not self._offline_collection:
            raise RuntimeError(
                "DGLSMobiWavePolicy requires offline initialization: call "
                "begin_offline_training(), collect a historical stream, and "
                "call finalize_offline_training() before evaluation"
            )
        if (
            env.time == 0
            and self._controller is not None
            and self.feature_builder.last_time is not None
            and self.feature_builder.last_time >= 0
        ):
            self.start_test_stream()
        if self._env is not env:
            self._stream_serial += 1
            phase = "offline" if self._offline_collection else "test"
            self._active_stream_id = (
                f"{phase}:{self._stream_serial}:seed={int(env.config.seed)}"
            )
        self._env = env
        if (
            not self._offline_collection
            and self._historical_od_prior is not None
            and env.time == 0
        ):
            env.set_historical_od_prior(self._historical_od_prior)
        if self._controller is not None and env.time not in self._checkpoints:
            self._checkpoints[env.time] = EnvCheckpoint.capture(
                env.export_replay_state(),
                time=env.time,
                metadata={"seed": env.config.seed},
            )

    def observe_warmup_transition(self) -> None:
        """Capture the pre-step t_0 state after a common warm-up transition."""

        if self._env is None or self._controller is None:
            return
        time = int(self._env.time)
        if time not in self._checkpoints:
            self._checkpoints[time] = EnvCheckpoint.capture(
                self._env.export_replay_state(),
                time=time,
                metadata={"seed": self._env.config.seed},
            )

    def begin_offline_training(self) -> None:
        """Collect a causal training history before the evaluation stream."""

        if self._records or self._controller is not None:
            raise RuntimeError("offline training must start on a fresh MobiWave policy")
        self._offline_collection = True
        # Collect multinomial behavior actions without an unrecorded
        # scale-dropout mask in their stored behavior probabilities.
        self.model.eval()
        self.stochastic_dispatch = True

    def finalize_offline_training(self) -> None:
        """Fit theta*, initialize R/bandwidths, then open a fresh test stream."""

        if not self._offline_collection:
            raise RuntimeError("begin_offline_training must be called first")
        complete = [
            record
            for record in self._records
            if self._demand_target(record, self._records) is not None
        ]
        required = int(self.config.dgls_min_reference)
        if len(complete) < required:
            raise RuntimeError(
                f"offline history has {len(complete)} labeled steps; "
                f"{required} are required"
            )
        labeled_history = [
            self._materialize_record_labels(record, self._records)
            for record in complete
        ]
        labeled_history = self._with_resolved_advantages(labeled_history)
        if self._env is None:
            raise RuntimeError("offline history environment is unavailable")
        self._historical_od_prior = self._env.historical_od_counts()
        self._initialize_controller(
            training_records=labeled_history,
            reference_records=labeled_history,
        )
        self._offline_collection = False
        self.start_test_stream()

    def start_test_stream(self) -> None:
        """Retain the accepted state but reset all stream-local evidence."""

        if self._controller is None:
            raise RuntimeError("MobiWave has no accepted offline state")
        accepted = self._controller.accepted
        controller_config = self._controller.config
        self._load_parameter_state(self.model, accepted.model_state)
        self._optimizer = self._new_optimizer(
            self.model,
            self._layer_registry(self.model),
        )
        self._optimizer.load_state_dict(accepted.optimizer_state)
        self._importance = accepted.importance_state
        self._reference_records = accepted.reference_state["records"]
        self._controller = DGLSTransactionController(
            accepted,
            controller_config,
        )
        super().reset()
        self.model.eval()
        self.stochastic_dispatch = False
        self._env = None
        self._records = []
        self._pending = None
        self._checkpoints = {}
        self._builder_before = {}
        self._reward_mean = 0.0
        self._reward_count = 0
        self._last_attempt_time = -1
        self._validation_audit = {}
        self._runtime_events = []
        self._offline_collection = False
        self._reactivation_pending = True
        self._graph_record_cache = OrderedDict()

    def reset(self) -> None:
        super().reset()
        self._records = []
        self._pending = None
        self._checkpoints = {}
        self._builder_before = {}
        self._controller = None
        self._optimizer = None
        self._importance = None
        self._reference_records = []
        self._layer_bandwidths = {}
        self._reward_mean = 0.0
        self._reward_count = 0
        self._last_attempt_time = -1
        self._validation_audit = {}
        self._runtime_events = []
        self._offline_collection = False
        self._reactivation_pending = True
        self._stream_serial = 0
        self._record_serial = 0
        self._active_stream_id = "unbound:0"
        self._historical_od_prior = None
        self._spectral_reference_cache = OrderedDict()
        self._reference_match_source = None
        self._reference_match_source_length = 0
        self._reference_match_metadata_cache = None
        self._graph_record_cache = OrderedDict()

    def export_state(
        self,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Cannot checkpoint MobiWave during an unfinished live step")
        if self._offline_collection:
            raise RuntimeError("Cannot checkpoint an unfinished offline-training phase")
        controller_state = (
            None
            if self._controller is None
            else self._controller.state_dict()
        )
        payload = {
            "format_version": 3,
            "base_policy": super().export_state(),
            "records": copy.deepcopy(self._records),
            "checkpoints": copy.deepcopy(self._checkpoints),
            "builder_before": copy.deepcopy(self._builder_before),
            "layer_bandwidths": dict(self._layer_bandwidths),
            "reward_mean": self._reward_mean,
            "reward_count": self._reward_count,
            "last_attempt_time": self._last_attempt_time,
            "validation_audit": copy.deepcopy(self._validation_audit),
            "runtime_events": copy.deepcopy(self._runtime_events),
            "reactivation_pending": self._reactivation_pending,
            "stream_serial": self._stream_serial,
            "record_serial": self._record_serial,
            "active_stream_id": self._active_stream_id,
            "historical_od_prior": (
                None
                if self._historical_od_prior is None
                else self._historical_od_prior.copy()
            ),
            "controller": controller_state,
            "spectral_bandwidths": (
                ()
                if self._controller is None
                else tuple(self._controller.config.bandwidths)
            ),
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
            payload = dict(source)
        if "base_policy" not in payload:
            super().load_exported_state(
                payload,
                map_location=map_location or self.device,
            )
            return
        format_version = int(payload.get("format_version", -1))
        if format_version not in {2, 3}:
            raise ValueError("unsupported DGLS MobiWave policy state format")
        super().load_exported_state(
            payload["base_policy"],
            map_location=map_location or self.device,
        )
        self._records = copy.deepcopy(payload["records"])
        self._checkpoints = copy.deepcopy(payload["checkpoints"])
        self._builder_before = copy.deepcopy(payload["builder_before"])
        if format_version == 2:
            self._reference_records = copy.deepcopy(
                payload["reference_records"]
            )
            self._importance = copy.deepcopy(payload["importance"])
        else:
            self._reference_records = []
            self._importance = None
        self._layer_bandwidths = {
            str(name): float(value)
            for name, value in payload["layer_bandwidths"].items()
        }
        self._reward_mean = float(payload["reward_mean"])
        self._reward_count = int(payload["reward_count"])
        self._last_attempt_time = int(payload["last_attempt_time"])
        self._validation_audit = copy.deepcopy(payload["validation_audit"])
        self._runtime_events = copy.deepcopy(payload["runtime_events"])
        self._reactivation_pending = bool(
            payload.get("reactivation_pending", True)
        )
        self._stream_serial = int(payload.get("stream_serial", 0))
        self._record_serial = int(
            payload.get(
                "record_serial",
                max(
                    (
                        int(record.get("record_sequence", 0))
                        for record in (
                            *self._reference_records,
                            *self._records,
                        )
                    ),
                    default=0,
                ),
            )
        )
        self._active_stream_id = str(
            payload.get("active_stream_id", "restored:0")
        )
        raw_od_prior = payload.get("historical_od_prior")
        self._historical_od_prior = (
            None
            if raw_od_prior is None
            else np.asarray(raw_od_prior, dtype=np.float64).copy()
        )
        self._spectral_reference_cache = OrderedDict()
        self._reference_match_source = None
        self._reference_match_source_length = 0
        self._reference_match_metadata_cache = None
        self._graph_record_cache = OrderedDict()
        self._pending = None
        self._env = None
        controller_payload = payload.get("controller")
        if controller_payload is None:
            self._controller = None
            self._optimizer = None
            return
        accepted_payload = dict(controller_payload["accepted"])
        accepted_payload["model_state"] = (
            self.model._migrate_legacy_parameter_state(
                accepted_payload["model_state"]
            )
        )
        controller_payload = {
            **dict(controller_payload),
            "accepted": accepted_payload,
        }
        accepted = AcceptedState(
            model_state=accepted_payload["model_state"],
            optimizer_state=accepted_payload["optimizer_state"],
            importance_state=accepted_payload["importance_state"],
            reference_state=accepted_payload["reference_state"],
            version=int(accepted_payload["version"]),
        )
        self._controller = DGLSTransactionController(
            accepted,
            self._controller_config(payload["spectral_bandwidths"]),
        )
        self._controller.load_state_dict(controller_payload)
        restored = self._controller.materialize_accepted()
        self._load_parameter_state(self.model, restored.model_state)
        registry = self._layer_registry(self.model)
        self._optimizer = self._new_optimizer(self.model, registry)
        self._optimizer.load_state_dict(restored.optimizer_state)
        self._importance = restored.importance_state
        self._reference_records = restored.reference_state["records"]

    def decide(self, snapshot: Mapping[str, Any]):
        time = int(snapshot["time"])
        self._credit_predecision_cancellations(snapshot)
        # Algorithm 1 first encodes the current causal state with the accepted
        # model.  Drift detection may then build and validate an isolated
        # candidate; an accepted candidate re-dispatches this same state.
        needs_replay = self._controller is not None
        builder_before = (
            self.feature_builder.state_dict()
            if needs_replay
            else None
        )
        decision = super().decide(snapshot)
        if builder_before is not None:
            self._builder_before[time] = builder_before
        if self.last_outputs is None or self.last_edge_allocation is None:
            self._pending = None
            return decision

        pending = self._pending_from_snapshot(snapshot)
        should_attempt = (
            self._controller is not None
            and time - self._last_attempt_time >= int(self.config.dgls_attempt_interval)
        )
        if should_attempt:
            accepted = self._attempt_adaptation(time, pending.record)
            self._last_attempt_time = time
            if accepted:
                # The live model has changed atomically.  Recompute U_t with
                # the accepted state before executing the current decision.
                decision = super().decide(snapshot)
                if self.last_outputs is None or self.last_edge_allocation is None:
                    raise RuntimeError("accepted MobiWave model failed to re-dispatch")
                pending = self._pending_from_snapshot(snapshot)
        self._pending = pending
        return decision

    def _pending_from_snapshot(
        self,
        snapshot: Mapping[str, Any],
    ) -> _PendingTransition:
        """Materialize the behavior-policy tuple for the current causal step."""

        time = int(snapshot["time"])
        if (
            self.last_outputs is None
            or self.last_edge_allocation is None
            or self.last_features is None
            or self.last_available_supply is None
            or self.last_backlog is None
            or self.last_current_demand is None
            or self.last_external_context is None
        ):
            raise RuntimeError("MobiWave has no executable output for this step")
        available = self.last_available_supply.astype(np.int64, copy=True)
        outputs = self.last_outputs
        old_log_probability = float(
            self.model.flow_log_probability(
                outputs["edge_probabilities"],
                self.last_edge_allocation.to(self.device),
            )
            .detach()
            .cpu()
            .item()
        )
        feasible = {
            int(source): tuple(int(value) for value in destinations)
            for source, destinations in snapshot["feasible_destinations"].items()
        }
        edge_travel = {
            (int(source), int(destination)): float(value)
            for (source, destination), value in snapshot["edge_travel_times"].items()
        }
        edge_cost = {
            edge: float(self.config.vehicle_move_cost_per_grid)
            * self.network.hex_distance(*edge)
            for edge in edge_travel
        }
        edge_graph_weights = {
            (int(source), int(destination)): float(value)
            for (source, destination), value in snapshot.get(
                "edge_graph_weights",
                {},
            ).items()
        }
        topology_signature = tuple(
            (
                source,
                destination,
                float(
                    0.0
                    if source == destination
                    else edge_graph_weights.get(
                        (source, destination),
                        1.0,
                    )
                ),
            )
            for source, destinations in sorted(feasible.items())
            for destination in destinations
        )
        cached_graph = self._graph_record_cache.get(topology_signature)
        if cached_graph is None:
            self._graph_record_cache[topology_signature] = (
                feasible,
                edge_cost,
                edge_graph_weights,
                topology_signature,
            )
            while len(self._graph_record_cache) > 64:
                self._graph_record_cache.popitem(last=False)
        else:
            self._graph_record_cache.move_to_end(topology_signature)
            (
                feasible,
                edge_cost,
                edge_graph_weights,
                topology_signature,
            ) = cached_graph
        travel_signature = tuple(
            (
                source,
                destination,
                edge_travel[(source, destination)],
            )
            for source, destination, _ in topology_signature
        )
        self._record_serial += 1
        record = {
            "time": time,
            "stream_id": self._active_stream_id,
            "record_uid": f"{self._active_stream_id}:time={time}",
            "record_sequence": self._record_serial,
            "features": self.last_features.copy(),
            "available": available,
            "backlog": self.last_backlog.astype(np.float32, copy=True),
            "observed_demand": self.last_current_demand.astype(
                np.float32,
                copy=True,
            ),
            "external_context": self.last_external_context.astype(
                np.float32,
                copy=True,
            ),
            "new_demand": np.asarray(
                snapshot["zone_new_requests"], dtype=np.float32
            ).copy(),
            "feasible_destinations": feasible,
            "edge_costs": edge_cost,
            "edge_graph_weights": edge_graph_weights,
            "graph_version": str(snapshot["graph_version"]),
            "flow": {
                (int(source), int(destination)): int(count)
                for source, destination, count in zip(
                    self.model.edge_sources.detach().cpu().tolist(),
                    self.model.edge_destinations.detach().cpu().tolist(),
                    self.last_edge_allocation[0].detach().cpu().tolist(),
                )
                if int(count) > 0
            },
            "old_log_probability": old_log_probability,
            "behavior_stochastic": bool(
                self.model.training
                if self.stochastic_dispatch is None
                else self.stochastic_dispatch
            ),
            "advantage": 0.0,
            "reward": 0.0,
            "edge_return_targets": {},
            "edge_vehicle_ids": {
                edge: tuple(
                    sorted(
                        vehicle_id
                        for vehicle_id, vehicle_edge in self.last_vehicle_edges.items()
                        if vehicle_edge == edge
                    )
                )
                for edge in sorted(set(self.last_vehicle_edges.values()))
            },
            "vehicle_rewards": {},
            # These cancellations were observed at the current pre-decision
            # boundary.  They are retained even before U_t is executed so
            # older H-step labels can mature causally at decision time t.
            "canceled_by_zone": {
                int(zone): int(count)
                for zone, count in snapshot.get(
                    "canceled_by_zone",
                    {},
                ).items()
            },
            "request_signature": self._request_signature(snapshot["new_requests"]),
            "topology_signature": topology_signature,
            "travel_signature": travel_signature,
        }
        return _PendingTransition(
            record=record,
            vehicle_edges=dict(self.last_vehicle_edges),
        )

    def observe_transition(self, snapshot, decision, result) -> None:
        del snapshot, decision
        if self._pending is not None:
            edge_values: dict[tuple[int, int], list[float]] = {}
            for vehicle_id, edge in self._pending.vehicle_edges.items():
                if vehicle_id in result.vehicle_net_rewards:
                    edge_values.setdefault(edge, []).append(
                        float(result.vehicle_net_rewards[vehicle_id])
                    )
            self._pending.record["edge_return_targets"] = {
                edge: float(np.mean(values)) for edge, values in edge_values.items()
            }
            self._pending.record["vehicle_rewards"] = {
                int(vehicle_id): float(value)
                for vehicle_id, value in result.vehicle_net_rewards.items()
            }
            self._pending.record["canceled_by_zone"] = {
                int(zone): int(count)
                for zone, count in result.canceled_by_zone.items()
            }
            # The cancellations in this result occurred before U_t.  Remove
            # them from the current action reward; decide(t+1) credits them to
            # U_t after that outcome becomes observable.
            action_reward = float(result.net_reward) + (
                float(self.config.cancel_penalty) * float(result.canceled)
            )
            advantage = action_reward - self._reward_mean
            self._pending.record["advantage"] = advantage
            self._pending.record["reward"] = action_reward
            self._records.append(self._pending.record)
            self._pending = None
            self._reward_count += 1
            self._reward_mean += (
                action_reward - self._reward_mean
            ) / self._reward_count
            live_capacity = max(
                int(self.config.dgls_reference_capacity)
                + self.model.forecast_horizon,
                int(self.config.dgls_train_window)
                + int(self.config.dgls_validation_window)
                + self.model.forecast_horizon
                + 4,
            )
            if not self._offline_collection and len(self._records) > live_capacity:
                self._records = self._records[-live_capacity:]

        if self._env is not None and self._controller is not None:
            time = int(self._env.time)
            self._checkpoints[time] = EnvCheckpoint.capture(
                self._env.export_replay_state(),
                time=time,
                metadata={"seed": self._env.config.seed},
            )
            retain_from = max(
                0,
                time
                - int(self.config.dgls_train_window)
                - int(self.config.dgls_validation_window)
                - self.model.forecast_horizon
                - 4,
            )
            self._checkpoints = {
                key: value for key, value in self._checkpoints.items() if key >= retain_from
            }
            self._builder_before = {
                key: value for key, value in self._builder_before.items() if key >= retain_from
            }

    def _initialize_controller(
        self,
        *,
        training_records: Sequence[dict[str, Any]],
        reference_records: Sequence[dict[str, Any]],
    ) -> None:
        if self._controller is not None:
            raise RuntimeError("DGLS controller is already initialized")
        capacity = int(self.config.dgls_reference_capacity)
        self._pretrain(training_records)
        self._reference_records = self._compact_reference_records(
            reference_records,
            capacity,
        )
        registry = self._layer_registry(self.model)
        self._optimizer = self._new_optimizer(self.model, registry)
        importance_values = {
            parameter_name: torch.zeros_like(parameter.detach().cpu())
            for parameter_name, parameter in self._adaptable_named_parameters(
                self.model
            )
        }
        self._importance = {
            "accepted_count": 0,
            "values": importance_values,
        }
        spectral_bandwidths, self._layer_bandwidths = self._calibrate_bandwidths(
            self.model,
            self._reference_records,
        )
        controller_config = self._controller_config(spectral_bandwidths)
        accepted = AcceptedState(
            model_state=self._parameter_state(self.model),
            optimizer_state=self._optimizer.state_dict(),
            importance_state=self._importance,
            reference_state={
                "records": self._reference_records,
                "capacity": capacity,
            },
            version=0,
        )
        self._controller = DGLSTransactionController(accepted, controller_config)

    def _controller_config(
        self,
        spectral_bandwidths: Sequence[float],
    ) -> ControllerConfig:
        return ControllerConfig(
            bandwidths=tuple(float(value) for value in spectral_bandwidths),
            threshold_on=float(self.config.dgls_threshold_on),
            threshold_off=float(self.config.dgls_threshold_off),
            budget_ratio=float(self.config.dgls_budget_ratio),
            mmd_sample_cap=int(self.config.dgls_mmd_sample_cap),
            persistence_decay=float(self.config.dgls_beta_persistence),
            eta_base=float(self.config.dgls_base_lr),
            eta_min=float(self.config.dgls_base_lr),
            eta_max=float(self.config.dgls_base_lr),
            eta_shock_scale=0.0,
            eta_persistence_scale=0.0,
            weight_activation=float(self.config.dgls_activation_score_weight),
            weight_gradient=float(self.config.dgls_gradient_score_weight),
            weight_importance=float(self.config.dgls_importance_score_weight),
            reward_margin=float(self.config.dgls_reward_margin),
            required_violations=REQUIRED_VIOLATIONS,
            violation_tolerances={
                "maximum_wait_breaches": float(self.config.dgls_wait_tolerance),
                "infeasible_moves": float(self.config.dgls_infeasible_tolerance),
                "zone_service_shortfall": float(
                    self.config.dgls_service_shortfall_tolerance
                ),
            },
        )

    def _attempt_adaptation(
        self,
        decision_time: int,
        current_evidence: Mapping[str, Any],
    ) -> bool:
        if self._controller is None or self._optimizer is None:
            return False
        needed = int(self.config.dgls_train_window) + int(
            self.config.dgls_validation_window
        )
        past = [record for record in self._records if int(record["time"]) < decision_time]
        if len(past) < needed:
            return False
        recent = past[-needed:]
        split = CausalSplit(
            adaptation=recent[: int(self.config.dgls_train_window)],
            validation=recent[int(self.config.dgls_train_window) :],
            decision_time=decision_time,
        )
        accepted_reference = self._reference_records
        stable_model = copy.deepcopy(self.model)
        accepted_model_state = self._parameter_state(self.model)
        accepted_importance = self._importance
        if accepted_importance is None:
            raise RuntimeError("DGLS accepted importance state is unavailable")
        # C_t includes the current encoded state F_t, while both the
        # adaptation prefix and later validation suffix remain before t.
        drift_window = [*recent[1:], dict(current_evidence)]
        recent_features, matched_reference_features, gates = (
            self._matched_spectral_evidence(
                stable_model,
                drift_window,
                accepted_reference,
            )
        )
        behavior_records: list[dict[str, Any]] | None = None
        behavior_matches: list[list[int]] | None = None
        signals: LayerSignals | None = None

        def stochastic_training_records() -> list[dict[str, Any]]:
            nonlocal behavior_records
            if behavior_records is None:
                behavior_records = self._replay_stochastic_prefix(
                    split,
                    accepted_model_state,
                )
            return behavior_records

        def stochastic_training_matches() -> list[list[int]]:
            nonlocal behavior_matches
            if behavior_matches is None:
                behavior_matches = self._reference_record_indices(
                    stochastic_training_records(),
                    accepted_reference,
                )
            return behavior_matches

        def layer_signal_source() -> LayerSignals:
            nonlocal signals
            if signals is None:
                training_records = stochastic_training_records()
                signals = self._layer_signals(
                    stable_model,
                    training_records,
                    accepted_reference,
                    accepted_importance,
                    all_records=training_records,
                    match_indices=stochastic_training_matches(),
                )
            return signals

        attempt_started = wall_time.perf_counter()
        self._validation_audit = {}
        was_active = bool(self._controller.dynamics.active)
        transaction = self._controller.propose(
            split=split,
            recent_features=recent_features,
            reference_features=matched_reference_features,
            recent_gate_weights=gates,
            layer_signals=layer_signal_source,
            candidate_callback=lambda state, causal, context: self._build_candidate(
                state,
                causal,
                context,
                accepted_reference,
                self._reactivation_pending,
                stochastic_training_records(),
                [*self._records, dict(current_evidence)],
                stochastic_training_matches(),
            ),
        )
        if not self._controller.dynamics.active:
            self._reactivation_pending = True
        elif not was_active:
            self._reactivation_pending = True
        if transaction is None:
            return False
        total_layer_cost = sum(layer_signal_source().costs.values())
        staged_live = None

        def validate_and_stage(stable, candidate, records):
            nonlocal staged_live
            staged_live = self._stage_live_state(candidate)
            return self._validate_transaction(stable, candidate, records)

        result = self._controller.finalize(
            transaction,
            validate_and_stage,
        )
        if result.accepted:
            if staged_live is None:
                raise RuntimeError(
                    "accepted DGLS candidate has no staged live state"
                )
            (
                self.model,
                self._optimizer,
                self._importance,
                self._reference_records,
            ) = staged_live
            self._spectral_reference_cache.clear()
            self._reference_match_source = None
            self._reference_match_source_length = 0
            self._reference_match_metadata_cache = None
            self._reactivation_pending = False
        self._runtime_events.append(
            {
                "event_type": "adaptation_runtime",
                "decision_time": decision_time,
                "accepted": result.accepted,
                "reason": result.reason,
                "adaptation_latency_ms": (
                    wall_time.perf_counter() - attempt_started
                )
                * 1000.0,
                "updated_parameter_count": int(
                    transaction.context.selection.selected_cost
                ),
                "updated_parameter_ratio": float(
                    transaction.context.selection.selected_cost
                    / max(1.0, total_layer_cost)
                ),
                **copy.deepcopy(self._validation_audit),
            }
        )
        return bool(result.accepted)

    def _stage_live_state(
        self,
        candidate,
    ) -> tuple[
        MobiWaveDispatchNet,
        DGLSFastSlowOptimizer,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        """Validate all externally deployed state before atomic reference swap."""

        importance_state = candidate.importance_state
        reference_state = candidate.reference_state
        staged_model = copy.deepcopy(self.model)
        self._load_parameter_state(staged_model, candidate.model_state)
        staged_optimizer = self._new_optimizer(
            staged_model,
            self._layer_registry(staged_model),
        )
        staged_optimizer.load_state_dict(candidate.optimizer_state)
        return (
            staged_model,
            staged_optimizer,
            importance_state,
            reference_state["records"],
        )

    def _build_candidate(
        self,
        state: MutableState,
        split: CausalSplit,
        context,
        reference: Sequence[dict[str, Any]],
        reactivated: bool,
        training_records: Sequence[dict[str, Any]],
        label_context: Sequence[dict[str, Any]],
        match_indices: Sequence[Sequence[int]],
    ) -> MutableState:
        candidate = copy.deepcopy(self.model)
        self._load_parameter_state(candidate, state.model_state)
        registry = self._layer_registry(candidate)
        optimizer = self._new_optimizer(candidate, registry)
        optimizer.load_state_dict(state.optimizer_state)
        if reactivated:
            optimizer.deactivate()
        eta_by_layer = self._effective_layer_rates(
            context.selection,
            context.eta,
        )
        optimizer.configure(
            context.selection.selected,
            eta_by_layer=eta_by_layer,
            shock=context.dynamics.S,
            persistence=context.dynamics.P,
            fixed_schedule=False,
        )
        selected_layers = tuple(context.selection.selected)
        selected_parameter_order = tuple(
            registry_parameters(registry, selected_layers)
        )
        selected_parameter_names = set(selected_parameter_order)
        candidate_parameters = dict(candidate.named_parameters())
        for parameter_name, parameter in candidate_parameters.items():
            parameter.requires_grad_(parameter_name in selected_parameter_names)
        stable_parameters = {
            name: state.model_state[name].to(self.device)
            for name in selected_parameter_order
        }
        stable_reference_means = self._matched_reference_activation_means(
            candidate,
            training_records,
            reference,
            match_indices=match_indices,
            layers=selected_layers,
        )
        importance = state.importance_state
        importance_hat = self._importance_hat(importance)
        parameter_importance = {
            name: importance_hat.get(
                name,
                torch.zeros_like(candidate_parameters[name].detach().cpu()),
            ).to(candidate_parameters[name].device)
            for name in selected_parameter_order
        }
        activation_sources = {
            layer: self._activation_key(layer)
            for layer in selected_layers
        }
        selected_parameters = [
            candidate_parameters[name]
            for name in selected_parameter_order
        ]

        # Online adaptation uses deterministic scale evidence.  Keeping the
        # candidate in eval mode disables scale dropout without disabling
        # gradients, so the activation anchor measures drift rather than
        # independently sampled masks.
        candidate.eval()
        with self._isolated_torch_rng():
            torch.manual_seed(
                int(self.config.seed) + int(context.decision_time) * 1009
            )
            for _ in range(int(self.config.dgls_inner_steps)):
                optimizer.zero_grad()
                recent_losses = self._loss_on_records(
                    candidate,
                    training_records,
                    training_records,
                    include_policy=True,
                    activation_sources=activation_sources,
                )
                candidate_activation_means = recent_losses[
                    "activation_means"
                ]
                anchor = recent_losses["total"] * 0.0
                for layer in selected_layers:
                    if (
                        layer in candidate_activation_means
                        and layer in stable_reference_means
                    ):
                        anchor = anchor + torch.sum(
                            (
                                candidate_activation_means[layer]
                                - stable_reference_means[layer].detach()
                            )
                            ** 2
                        )
                protection = recent_losses["total"] * 0.0
                for parameter_name in selected_parameter_order:
                    parameter = candidate_parameters[parameter_name]
                    protection = protection + torch.sum(
                        parameter_importance[parameter_name]
                        * (
                            parameter
                            - stable_parameters[parameter_name]
                        )
                        ** 2
                    )
                loss = (
                    recent_losses["total"]
                    + float(self.config.dgls_activation_loss_weight) * anchor
                    + float(self.config.dgls_importance_loss_weight) * protection
                )
                loss.backward()
                # Accumulate the identically normalized matched-reference
                # gradient one historical record at a time.  This is
                # mathematically the same objective as forming one large
                # reference loss, while each forward graph can be released
                # immediately instead of retaining the full buffer in memory.
                self._backward_matched_reference_loss(
                    candidate,
                    training_records,
                    reference,
                    match_indices=match_indices,
                )
                torch.nn.utils.clip_grad_norm_(
                    selected_parameters,
                    5.0,
                )
                optimizer.step()

        for parameter in candidate.parameters():
            parameter.requires_grad_(True)
        validation_records = split.validation
        validation_behavior = self._replay_stochastic_window(
            validation_records,
            state.model_state,
            decision_time=split.decision_time,
            rng_offset=29,
        )
        staged_importance = self._updated_importance(
            candidate,
            validation_behavior,
            importance,
        )
        staged_reference = list(state.reference_state["records"])
        staged_reference.extend(
            self._materialize_record_labels(record, label_context)
            for record in validation_records
        )
        capacity = int(state.reference_state["capacity"])
        staged_reference = self._compact_reference_records(
            staged_reference,
            capacity,
        )
        state.model_state = self._parameter_state(candidate)
        state.optimizer_state = optimizer.state_dict()
        state.importance_state = staged_importance
        state.reference_state = {
            "records": staged_reference,
            "capacity": capacity,
        }
        return state

    def _backward_matched_reference_loss(
        self,
        model: MobiWaveDispatchNet,
        recent: Sequence[dict[str, Any]],
        reference: Sequence[dict[str, Any]],
        *,
        match_indices: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Accumulate the exactly normalized matched-reference gradient.

        Graph-compatible reference records are differentiated in bounded
        batches so their computation graphs do not fill accelerator memory.
        Repeated matches are represented by integer multiplicities, preserving
        the objective used by the direct mean over all matched terms.
        """

        reference_weight = float(self.config.dgls_reference_loss_weight)
        if reference_weight == 0.0:
            return
        if match_indices is None:
            match_indices = self._reference_record_indices(recent, reference)
        zone_multiplicities: dict[int, dict[int, int]] = {}
        for row in match_indices:
            for zone, reference_index in enumerate(row):
                counts = zone_multiplicities.setdefault(reference_index, {})
                counts[zone] = counts.get(zone, 0) + 1

        demand_count = 0
        return_count = 0
        record_edges: dict[int, tuple[tuple[int, int], ...]] = {}
        for reference_index, zone_counts in zone_multiplicities.items():
            record = reference[reference_index]
            demand_target = record.get("resolved_demand_target")
            if demand_target is not None:
                demand_count += sum(zone_counts.values())
            edge_targets = record.get("resolved_edge_return_targets", {})
            feasible = record["feasible_destinations"]
            edges = tuple(
                (source, destination)
                for source in range(self.network.zone_count)
                for destination in sorted(
                    {int(value) for value in feasible[source]}
                )
            )
            record_edges[reference_index] = edges
            return_count += sum(
                zone_counts.get(source, 0)
                for source, destination in edges
                if (source, destination) in edge_targets
            )

        demand_weight = (
            reference_weight
            * float(self.config.mobiwave_demand_loss_weight)
            / max(1, demand_count)
        )
        return_weight = (
            reference_weight
            * float(self.config.mobiwave_return_loss_weight)
            / max(1, return_count)
        )
        selected_indices = sorted(zone_multiplicities)
        for batch_indices in self._topology_batches(
            reference,
            selected_indices,
        ):
            batch_records = [reference[index] for index in batch_indices]
            outputs = self._forward_record_batch(model, batch_records)
            batch_loss: torch.Tensor | None = None
            for batch_row, reference_index in enumerate(batch_indices):
                record = reference[reference_index]
                demand_target = record.get("resolved_demand_target")
                if demand_target is not None and demand_count:
                    demand_values = np.asarray(demand_target)
                    for zone, multiplicity in zone_multiplicities[
                        reference_index
                    ].items():
                        target = torch.as_tensor(
                            demand_values[zone],
                            dtype=outputs["demand"].dtype,
                            device=self.device,
                        )
                        contribution = (
                            demand_weight
                            * multiplicity
                            * torch.nn.functional.huber_loss(
                                outputs["demand"][batch_row, zone],
                                target,
                                reduction="mean",
                                delta=1.0,
                            )
                        )
                        batch_loss = (
                            contribution
                            if batch_loss is None
                            else batch_loss + contribution
                        )

                edge_targets = record.get(
                    "resolved_edge_return_targets",
                    {},
                )
                if edge_targets and return_count:
                    for edge_index, edge in enumerate(
                        record_edges[reference_index]
                    ):
                        multiplicity = zone_multiplicities[
                            reference_index
                        ].get(edge[0], 0)
                        if not multiplicity or edge not in edge_targets:
                            continue
                        target = outputs["edge_returns"].new_tensor(
                            float(edge_targets[edge])
                        )
                        contribution = (
                            return_weight
                            * multiplicity
                            * torch.nn.functional.huber_loss(
                                outputs["edge_returns"][
                                    batch_row,
                                    edge_index,
                                ],
                                target,
                                reduction="mean",
                                delta=1.0,
                            )
                        )
                        batch_loss = (
                            contribution
                            if batch_loss is None
                            else batch_loss + contribution
                        )
            if batch_loss is not None and batch_loss.requires_grad:
                batch_loss.backward()

    def _validate_transaction(self, stable, candidate, validation_records):
        records = tuple(validation_records)
        start = int(records[0]["time"])
        end = int(records[-1]["time"]) + 1
        checkpoint = self._checkpoints.get(start)
        builder_state = self._builder_before.get(start)
        if checkpoint is None or builder_state is None:
            raise RuntimeError("paired suffix checkpoint is unavailable")
        tape = ExogenousTape.capture(
            start_time=start,
            end_time=end,
            requests=tuple(record["request_signature"] for record in records),
            travel_times=tuple(record["travel_signature"] for record in records),
            topology=tuple(record["topology_signature"] for record in records),
            metadata={"graph_versions": tuple(record["graph_version"] for record in records)},
        )
        validator = PairedSuffixValidator(
            lambda checkpoint_value, model_state, tape_value: self._replay_suffix(
                checkpoint_value,
                model_state,
                tape_value,
                builder_state,
            ),
            required_violations=REQUIRED_VIOLATIONS,
            reward_margin=float(self.config.dgls_reward_margin),
            violation_tolerances={
                "maximum_wait_breaches": float(self.config.dgls_wait_tolerance),
                "infeasible_moves": float(self.config.dgls_infeasible_tolerance),
                "zone_service_shortfall": float(
                    self.config.dgls_service_shortfall_tolerance
                ),
            },
        )
        paired = validator.validate(
            checkpoint,
            stable.model_state,
            candidate.model_state,
            tape,
        )
        self._validation_audit = {
            "paired_replay_available": paired.available,
            "initial_state_hash": checkpoint.initial_state_hash,
            "request_hash": tape.request_hash,
            "travel_hash": tape.travel_hash,
            "topology_hash": tape.topology_hash,
        }
        if not paired.available or paired.stable is None or paired.candidate is None:
            raise RuntimeError(f"paired replay unavailable: {paired.reason}: {paired.error}")
        return ValidationPair(
            stable=paired.stable.candidate_metrics(),
            candidate=paired.candidate.candidate_metrics(),
        )

    def _replay_suffix(
        self,
        checkpoint: EnvCheckpoint,
        model_state: Mapping[str, torch.Tensor],
        tape: ExogenousTape,
        builder_state: Mapping[str, Any],
    ) -> ReplayOutcome:
        if self._env is None:
            raise RuntimeError("no environment is bound for paired replay")
        replay_env = DispatchEnv(
            self._env.config,
            scenario=self._env.scenario,
            record_detailed_logs=False,
        )
        replay_env.import_replay_state(checkpoint.restore_state())
        replay_policy = MobiWavePolicy(
            self.config,
            self.network,
            device=self.device,
        )
        replay_policy.feature_builder.load_state_dict(builder_state)
        self._load_parameter_state(replay_policy.model, model_state)
        replay_policy.set_eval()

        expected_requests = tuple(tape.restore_requests())
        expected_travel = tuple(tape.restore_travel_times())
        expected_topology = tuple(tape.restore_topology())
        reward = 0.0
        wait_breaches = 0.0
        infeasible = 0.0
        shortfall = np.zeros(self.network.zone_count, dtype=float)
        demand_total = np.zeros(self.network.zone_count, dtype=float)

        class ReplayMonitor:
            def __init__(self, policy):
                self.policy = policy
                self.index = 0
                self.step_shortfall = np.zeros(
                    policy.network.zone_count,
                    dtype=float,
                )
                self.step_demand = np.zeros(
                    policy.network.zone_count,
                    dtype=float,
                )

            def bind_environment(self, env):
                del env

            def decide(self, snapshot):
                index = self.index
                if DGLSMobiWavePolicy._request_signature(
                    snapshot["new_requests"]
                ) != expected_requests[index]:
                    raise RuntimeError("request tape mismatch")
                topology = tuple(
                    (
                        source,
                        destination,
                        float(
                            0.0
                            if source == destination
                            else snapshot.get("edge_graph_weights", {}).get(
                                (source, destination),
                                1.0,
                            )
                        ),
                    )
                    for source, destinations in sorted(
                        snapshot["feasible_destinations"].items()
                    )
                    for destination in destinations
                )
                if topology != expected_topology[index]:
                    raise RuntimeError("topology tape mismatch")
                travel = tuple(
                    (source, destination, float(value))
                    for (source, destination), value in sorted(
                        snapshot["edge_travel_times"].items()
                    )
                )
                if travel != expected_travel[index]:
                    raise RuntimeError("travel-time tape mismatch")
                decision = self.policy.decide(snapshot)
                served_by_zone = np.zeros(self.policy.network.zone_count, dtype=float)
                for vehicle_id, request_id in decision.assignments.items():
                    del vehicle_id
                    request = snapshot["requests"].get(request_id)
                    if request is not None:
                        served_by_zone[int(request.origin)] += 1.0
                demand = np.asarray(
                    snapshot["zone_open_requests"],
                    dtype=float,
                )
                self.step_shortfall = np.maximum(
                    0.0,
                    demand - served_by_zone,
                )
                self.step_demand = demand
                self.index += 1
                return decision

        monitor = ReplayMonitor(replay_policy)
        while replay_env.time < tape.end_time:
            before_events = len(replay_env.event_log)
            result = replay_env.step(monitor)
            reward += float(result.net_reward)
            wait_breaches += float(result.canceled)
            shortfall += monitor.step_shortfall
            demand_total += monitor.step_demand
            infeasible += sum(
                1.0
                for event in replay_env.event_log[before_events:]
                if event.get("event") == "invalid_action"
            )
        if monitor.index != len(expected_requests):
            raise RuntimeError("suffix replay length mismatch")
        # Outcomes that become observable immediately before the next
        # decision still belong to the held-out suffix's last action.
        _, boundary_canceled, _ = replay_env.advance_to_decision_boundary()
        reward -= float(self.config.cancel_penalty) * float(boundary_canceled)
        wait_breaches += float(boundary_canceled)
        zone_rates = np.divide(
            shortfall,
            demand_total,
            out=np.zeros_like(shortfall),
            where=demand_total > 0.0,
        )
        return ReplayOutcome(
            reward=reward,
            violations={
                "maximum_wait_breaches": wait_breaches,
                "infeasible_moves": infeasible,
                # Use the worst zone so improvements elsewhere cannot hide a
                # local service regression.
                "zone_service_shortfall": float(zone_rates.max(initial=0.0)),
            },
            initial_state_hash=checkpoint.initial_state_hash,
            request_hash=tape.request_hash,
            travel_hash=tape.travel_hash,
            topology_hash=tape.topology_hash,
        )

    def _replay_stochastic_prefix(
        self,
        split: CausalSplit,
        model_state: Mapping[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        """Generate the paper's multinomial behavior rollout for L_recent."""

        # Replay only the adaptation prefix.  Its trailing H-step labels stay
        # unresolved when they would cross into the held-out suffix, so no
        # validation outcome can enter the candidate-training objective.
        return self._replay_stochastic_window(
            split.adaptation,
            model_state,
            decision_time=split.decision_time,
            rng_offset=17,
            allow_terminal_labels=False,
        )

    def _replay_stochastic_window(
        self,
        source_records: Sequence[Mapping[str, Any]],
        model_state: Mapping[str, torch.Tensor],
        *,
        decision_time: int,
        rng_offset: int,
        allow_terminal_labels: bool = True,
    ) -> list[dict[str, Any]]:
        """Replay one causal window with a frozen multinomial behavior policy."""

        if self._env is None:
            raise RuntimeError("no environment is bound for behavior replay")
        source_records = tuple(source_records)
        if not source_records:
            raise ValueError("stochastic behavior replay requires records")
        start = int(source_records[0]["time"])
        end = int(source_records[-1]["time"]) + 1
        checkpoint = self._checkpoints.get(start)
        builder_state = self._builder_before.get(start)
        if checkpoint is None or builder_state is None:
            raise RuntimeError("stochastic-prefix checkpoint is unavailable")

        replay_env = DispatchEnv(
            self._env.config,
            scenario=self._env.scenario,
            record_detailed_logs=False,
        )
        replay_env.import_replay_state(checkpoint.restore_state())
        collector = DGLSMobiWavePolicy(
            self.config,
            self.network,
            device=self.device,
        )
        collector.begin_offline_training()
        collector.feature_builder.load_state_dict(builder_state)
        self._load_parameter_state(collector.model, model_state)

        with self._isolated_torch_rng():
            torch.manual_seed(
                int(self.config.seed)
                + int(decision_time) * 1009
                + int(rng_offset),
            )
            while replay_env.time < end:
                replay_env.step(collector)
            boundary_snapshot, _, _ = replay_env.advance_to_decision_boundary()
        collector._credit_predecision_cancellations(boundary_snapshot)

        records = copy.deepcopy(collector._records)
        expected_times = tuple(int(record["time"]) for record in source_records)
        actual_times = tuple(int(record["time"]) for record in records)
        if actual_times != expected_times:
            raise RuntimeError("stochastic-prefix time range mismatch")
        for replayed, expected in zip(records, source_records):
            if replayed["request_signature"] != expected["request_signature"]:
                raise RuntimeError("stochastic-prefix request tape mismatch")
            if replayed["travel_signature"] != expected["travel_signature"]:
                raise RuntimeError("stochastic-prefix travel tape mismatch")
            if replayed["topology_signature"] != expected["topology_signature"]:
                raise RuntimeError("stochastic-prefix topology tape mismatch")
        if records and allow_terminal_labels:
            terminal_canceled = {
                int(zone): int(count)
                for zone, count in boundary_snapshot.get(
                    "canceled_by_zone",
                    {},
                ).items()
            }
            terminal_demand = np.asarray(
                boundary_snapshot["zone_new_requests"],
                dtype=np.float32,
            ).copy()
            for record in records:
                record["terminal_boundary_time"] = int(end)
                record["terminal_canceled_by_zone"] = terminal_canceled
                record["terminal_new_demand"] = terminal_demand
        # Actions, rewards, demand labels, return labels, and clipped-policy advantages
        # must all come from this same stochastic replay.  Freezing the labels
        # here prevents accepted-importance updates from consulting the
        # deterministic live trajectory.
        records = [
            self._materialize_record_labels(record, records)
            for record in records
        ]
        return self._with_resolved_advantages(records)

    def _pretrain(self, records: Sequence[dict[str, Any]]) -> None:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.dgls_base_lr),
        )
        batch_size = int(self.config.mobiwave_pretrain_batch_size)
        self.model.train()
        for _ in range(int(self.config.mobiwave_pretrain_epochs)):
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                optimizer.zero_grad()
                losses = self._loss_on_records(
                    self.model,
                    batch,
                    self._records,
                    include_policy=True,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
        self.model.eval()

    def _loss_on_records(
        self,
        model: MobiWaveDispatchNet,
        records: Sequence[dict[str, Any]],
        all_records: Sequence[dict[str, Any]],
        *,
        include_policy: bool,
        activation_sources: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        zero = next(model.parameters()).sum() * 0.0
        if not records:
            empty: dict[str, Any] = {
                key: zero
                for key in ("total", "demand", "return", "policy", "gate_balance")
            }
            empty["activation_means"] = {}
            return empty
        policy_advantages: torch.Tensor | None = None
        if include_policy and records:
            times = [int(record["time"]) for record in records]
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("policy-loss records must be in chronological order")
            if all("resolved_advantage" in record for record in records):
                policy_advantages = torch.as_tensor(
                    [
                        float(record["resolved_advantage"])
                        for record in records
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                rewards = torch.as_tensor(
                    [float(record["reward"]) for record in records],
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.empty_like(rewards)
                running_return = rewards.new_zeros(())
                gamma = float(self.config.policy_discount)
                for index in range(len(records) - 1, -1, -1):
                    running_return = rewards[index] + gamma * running_return
                    returns[index] = running_return
                centered = returns - returns.mean()
                return_std = centered.square().mean().sqrt()
                policy_advantages = centered / (return_std + 1e-8)

        demand_sum = zero
        demand_count = 0
        return_sum = zero
        return_count = 0
        policy_terms: list[torch.Tensor] = []
        gate_rows: list[torch.Tensor] = []
        activation_sums: dict[str, torch.Tensor] = {}
        activation_counts: dict[str, int] = {}
        for batch_indices in self._topology_batches(
            records,
            list(range(len(records))),
        ):
            batch_records = [records[index] for index in batch_indices]
            outputs = self._forward_record_batch(model, batch_records)
            gate_values = outputs["gate_weights"]
            gate_rows.append(
                gate_values.reshape(-1, gate_values.shape[-1])
            )
            if activation_sources is not None:
                for layer, source in activation_sources.items():
                    activation = outputs["activations"][source]
                    flattened = activation.reshape(
                        -1,
                        activation.shape[-1],
                    )
                    contribution = flattened.sum(dim=0)
                    activation_sums[layer] = (
                        contribution
                        if layer not in activation_sums
                        else activation_sums[layer] + contribution
                    )
                    activation_counts[layer] = (
                        activation_counts.get(layer, 0)
                        + flattened.shape[0]
                    )

            edges = tuple(
                (int(source), int(destination))
                for source, destination in zip(
                    outputs["edge_sources"].detach().cpu().tolist(),
                    outputs["edge_destinations"].detach().cpu().tolist(),
                )
            )
            for batch_row, (record_index, record) in enumerate(
                zip(batch_indices, batch_records)
            ):
                demand_target = (
                    record.get("resolved_demand_target")
                    if "resolved_demand_target" in record
                    else self._demand_target(record, all_records)
                )
                edge_return_targets = (
                    record.get("resolved_edge_return_targets", {})
                    if "resolved_edge_return_targets" in record
                    else (
                        self._edge_return_targets(record, all_records)
                        if self._edge_return_window_complete(
                            record,
                            all_records,
                        )
                        else {}
                    )
                )
                if demand_target is not None:
                    target = torch.as_tensor(
                        demand_target,
                        dtype=outputs["demand"].dtype,
                        device=self.device,
                    )
                    demand_sum = demand_sum + torch.nn.functional.huber_loss(
                        outputs["demand"][batch_row],
                        target,
                        reduction="sum",
                        delta=1.0,
                    )
                    demand_count += int(target.numel())

                for edge_index, edge in enumerate(edges):
                    if edge not in edge_return_targets:
                        continue
                    target = outputs["edge_returns"].new_tensor(
                        float(edge_return_targets[edge])
                    )
                    return_sum = return_sum + torch.nn.functional.huber_loss(
                        outputs["edge_returns"][batch_row, edge_index],
                        target,
                        reduction="sum",
                        delta=1.0,
                    )
                    return_count += 1

            if not include_policy:
                continue
            stochastic_rows = [
                row
                for row, record in enumerate(batch_records)
                if bool(record.get("behavior_stochastic", True))
            ]
            if not stochastic_rows:
                # Online evaluation uses deterministic largest-remainder
                # allocation, so a multinomial behavior-policy ratio is not defined.
                continue
            # Behavior probabilities were recorded with scale dropout off.
            # Re-evaluate this graph-compatible batch in the same mode.
            if model.training:
                model.eval()
                policy_outputs = self._forward_record_batch(
                    model,
                    batch_records,
                )
                model.train()
            else:
                policy_outputs = outputs
            allocations: list[torch.Tensor] = []
            for record in batch_records:
                stored_flow = record["flow"]
                if isinstance(stored_flow, Mapping):
                    allocation = torch.as_tensor(
                        [
                            int(stored_flow.get(edge, 0))
                            for edge in edges
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    # Backward-compatible version-2 dense-flow records.
                    dense = torch.as_tensor(
                        stored_flow,
                        dtype=torch.long,
                        device=self.device,
                    )
                    allocation = dense[
                        policy_outputs["edge_sources"],
                        policy_outputs["edge_destinations"],
                    ]
                allocations.append(allocation)
            current_log_probabilities = model.flow_log_probability(
                policy_outputs["edge_probabilities"],
                torch.stack(allocations, dim=0),
            )
            for batch_row in stochastic_rows:
                record_index = batch_indices[batch_row]
                record = batch_records[batch_row]
                old_log_probability = (
                    current_log_probabilities.new_tensor(
                        float(record["old_log_probability"])
                    )
                )
                ratio = torch.exp(
                    (
                        current_log_probabilities[batch_row]
                        - old_log_probability
                    ).clamp(-20.0, 20.0)
                )
                clipped_ratio = ratio.clamp(
                    1.0 - float(self.config.policy_clip),
                    1.0 + float(self.config.policy_clip),
                )
                advantage = policy_advantages[record_index]
                policy_terms.append(
                    -torch.minimum(
                        ratio * advantage,
                        clipped_ratio * advantage,
                    )
                )

        demand_loss = demand_sum / max(1, demand_count)
        return_loss = return_sum / max(1, return_count)
        policy_loss = (
            torch.stack(policy_terms).mean() if policy_terms else zero
        )
        mean_gate = torch.cat(gate_rows, dim=0).mean(dim=0)
        gate_balance_loss = (
            (mean_gate - 1.0 / mean_gate.numel()) ** 2
        ).sum()
        total = (
            float(self.config.mobiwave_demand_loss_weight) * demand_loss
            + float(self.config.mobiwave_return_loss_weight) * return_loss
            + float(self.config.mobiwave_policy_loss_weight) * policy_loss
            + float(self.config.mobiwave_gate_balance_weight)
            * gate_balance_loss
        )
        return {
            "total": total,
            "demand": demand_loss,
            "return": return_loss,
            "policy": policy_loss,
            "gate_balance": gate_balance_loss,
            "activation_means": {
                layer: activation_sums[layer] / activation_counts[layer]
                for layer in activation_sums
            },
        }

    def _forward_record(
        self,
        model: MobiWaveDispatchNet,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._forward_record_batch(model, (record,))

    def _record_graph_signature(
        self,
        record: Mapping[str, Any],
    ) -> tuple[tuple[int, int, float], ...]:
        stored = record.get("topology_signature")
        if stored is not None:
            if isinstance(stored, tuple):
                return stored
            return tuple(
                (
                    int(source),
                    int(destination),
                    float(weight),
                )
                for source, destination, weight in stored
            )
        feasible = record["feasible_destinations"]
        weights = record.get("edge_graph_weights")
        return tuple(
            (
                source,
                destination,
                0.0
                if source == destination
                else float(
                    1.0
                    if weights is None
                    else weights.get((source, destination), 1.0)
                ),
            )
            for source in range(self.network.zone_count)
            for destination in sorted(
                {int(value) for value in feasible[source]}
            )
        )

    def _topology_batches(
        self,
        records: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        *,
        batch_size: int = 32,
    ) -> list[list[int]]:
        """Group record indices for exact batched graph forwards."""

        grouped: OrderedDict[
            tuple[tuple[int, int, float], ...],
            list[int],
        ] = OrderedDict()
        for index in indices:
            grouped.setdefault(
                self._record_graph_signature(records[index]),
                [],
            ).append(int(index))
        return [
            values[start : start + batch_size]
            for values in grouped.values()
            for start in range(0, len(values), batch_size)
        ]

    def _forward_record_batch(
        self,
        model: MobiWaveDispatchNet,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Forward records sharing one observed weighted road graph."""

        if not records:
            raise ValueError("record batch must be nonempty")
        graph_signature = self._record_graph_signature(records[0])
        if any(
            self._record_graph_signature(record) != graph_signature
            for record in records[1:]
        ):
            raise ValueError("record batch contains different graph topologies")
        model.update_graph(
            records[0]["feasible_destinations"],
            records[0].get("edge_graph_weights"),
        )
        edges = tuple(
            (source, destination)
            for source, destination, _ in graph_signature
        )
        def tensor_batch(
            key: str,
            *,
            default: np.ndarray | None = None,
        ) -> torch.Tensor:
            values = [
                np.asarray(
                    record.get(key, default)
                    if default is not None
                    else record[key],
                    dtype=np.float32,
                )
                for record in records
            ]
            if len(values) == 1:
                return torch.as_tensor(
                    values[0],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
            return torch.as_tensor(
                np.stack(values),
                dtype=torch.float32,
                device=self.device,
            )

        empty_external_context = np.zeros(
            (
                self.network.zone_count,
                model.EXTERNAL_CONTEXT_DIM,
            ),
            dtype=np.float32,
        )

        def travel_values(record: Mapping[str, Any]) -> list[float]:
            signature = record.get("travel_signature")
            if signature is not None:
                signature_edges = tuple(
                    (int(source), int(destination))
                    for source, destination, _ in signature
                )
                if signature_edges != edges:
                    raise ValueError(
                        "travel signature does not match graph topology"
                    )
                return [float(value) for _, _, value in signature]
            lookup = record["edge_travel_times"]
            return [
                float(lookup[(source, destination)])
                for source, destination in edges
            ]

        return model(
            tensor_batch("features"),
            available_supply=tensor_batch("available"),
            backlog=tensor_batch("backlog"),
            current_demand=tensor_batch("observed_demand"),
            external_context=tensor_batch(
                "external_context",
                default=empty_external_context,
            ),
            edge_travel_time=torch.as_tensor(
                [travel_values(record) for record in records],
                dtype=torch.float32,
                device=self.device,
            ),
            edge_move_cost=torch.as_tensor(
                [
                    [
                        record["edge_costs"][(source, destination)]
                        for source, destination in edges
                    ]
                    for record in records
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            time=int(records[0]["time"]),
            return_probability_matrix=False,
        )

    def _layer_signals(
        self,
        model: MobiWaveDispatchNet,
        recent: Sequence[dict[str, Any]],
        reference: Sequence[dict[str, Any]],
        importance: Mapping[str, Any],
        *,
        all_records: Sequence[dict[str, Any]] | None = None,
        match_indices: Sequence[Sequence[int]] | None = None,
    ) -> LayerSignals:
        registry = self._layer_registry(model)
        model.eval()
        recent_activations, reference_activations = self._matched_activation_sets(
            model,
            recent,
            reference,
            match_indices=match_indices,
        )
        activation = {
            layer: float(
                rbf_mmd2(
                    recent_activations[layer],
                    reference_activations[layer],
                    self._layer_bandwidths[layer],
                    sample_cap=int(self.config.dgls_mmd_sample_cap),
                )
                .detach()
                .cpu()
                .item()
            )
            for layer in registry
        }
        for parameter in model.parameters():
            parameter.grad = None
        losses = self._loss_on_records(
            model,
            recent,
            self._records if all_records is None else all_records,
            include_policy=True,
        )
        losses["total"].backward()
        named = dict(model.named_parameters())
        gradient = {}
        for layer, parameter_names in registry.items():
            squared = sum(
                float(torch.sum(named[name].grad.detach() ** 2).cpu().item())
                for name in parameter_names
                if named[name].grad is not None
            )
            cost = sum(named[name].numel() for name in parameter_names)
            gradient[layer] = math.sqrt(squared) / math.sqrt(max(1, cost))
        importance_hat = self._importance_hat(importance)
        layer_importance = {
            layer: float(
                sum(
                    float(importance_hat[name].float().sum().item())
                    for name in parameter_names
                )
                / max(
                    1,
                    sum(importance_hat[name].numel() for name in parameter_names),
                )
            )
            for layer, parameter_names in registry.items()
        }
        costs = {
            layer: float(sum(named[name].numel() for name in parameter_names))
            for layer, parameter_names in registry.items()
        }
        for parameter in model.parameters():
            parameter.grad = None
        return LayerSignals(
            activation=activation,
            gradient=gradient,
            importance=layer_importance,
            costs=costs,
        )

    def _matched_activation_sets(
        self,
        model: MobiWaveDispatchNet,
        recent: Sequence[dict[str, Any]],
        reference: Sequence[dict[str, Any]],
        *,
        match_indices: Sequence[Sequence[int]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Match node/edge activation units by time, zone, and local gap.

        Node layers use the same zone in the selected historical record.  Edge
        layers additionally use the same feasible move when it exists; a
        topology-changing stream falls back to the source-zone stay edge.
        """

        if not recent or not reference:
            raise ValueError("matched activation sets require nonempty inputs")
        registry = self._layer_registry(model)
        if match_indices is None:
            match_indices = self._reference_record_indices(recent, reference)
        cap = int(self.config.dgls_mmd_sample_cap)
        edge_layers = {"return_head", "policy_head"}

        def record_edges(record: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
            return tuple(
                (int(source), int(destination))
                for source, destinations in sorted(
                    record["feasible_destinations"].items()
                )
                for destination in sorted(
                    {int(value) for value in destinations}
                )
            )

        node_total = len(recent) * self.network.zone_count
        node_count = min(cap, node_total)
        node_positions = (
            ((2 * np.arange(node_count) + 1) * node_total)
            // (2 * node_count)
        )
        node_pairs: list[tuple[int, int, int]] = []
        for position in node_positions.tolist():
            recent_index, zone = divmod(
                int(position),
                self.network.zone_count,
            )
            node_pairs.append(
                (
                    recent_index,
                    zone,
                    match_indices[recent_index][zone],
                )
            )

        recent_edges = [record_edges(record) for record in recent]
        edge_total = sum(len(edges) for edges in recent_edges)
        edge_count = min(cap, edge_total)
        edge_positions = (
            ((2 * np.arange(edge_count) + 1) * edge_total)
            // (2 * edge_count)
        ).tolist()
        edge_pairs: list[
            tuple[int, int, int, int, int]
        ] = []
        position_index = 0
        offset = 0
        for recent_index, edges in enumerate(recent_edges):
            limit = offset + len(edges)
            while (
                position_index < len(edge_positions)
                and int(edge_positions[position_index]) < limit
            ):
                edge_index = int(edge_positions[position_index]) - offset
                source, destination = edges[edge_index]
                edge_pairs.append(
                    (
                        recent_index,
                        edge_index,
                        source,
                        destination,
                        match_indices[recent_index][source],
                    )
                )
                position_index += 1
            offset = limit

        node_requirements: dict[int, set[int]] = {}
        for _, zone, reference_index in node_pairs:
            node_requirements.setdefault(reference_index, set()).add(zone)
        edge_requirements: dict[int, set[tuple[int, int]]] = {}
        for _, _, source, destination, reference_index in edge_pairs:
            edge_requirements.setdefault(reference_index, set()).add(
                (source, destination)
            )

        reference_node_values: dict[
            tuple[str, int, int],
            torch.Tensor,
        ] = {}
        reference_edge_values: dict[
            tuple[str, int, int, int],
            torch.Tensor,
        ] = {}
        recent_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in registry
        }
        reference_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in registry
        }
        was_training = model.training
        model.eval()
        with torch.no_grad():
            selected_reference_indices = sorted(
                set(node_requirements) | set(edge_requirements)
            )
            for batch_indices in self._topology_batches(
                reference,
                selected_reference_indices,
            ):
                outputs = self._forward_record_batch(
                    model,
                    [reference[index] for index in batch_indices],
                )
                edges = tuple(
                    (int(source), int(destination))
                    for source, destination in zip(
                        model.edge_sources.detach().cpu().tolist(),
                        model.edge_destinations.detach().cpu().tolist(),
                    )
                )
                edge_lookup = {
                    edge: index for index, edge in enumerate(edges)
                }
                for batch_row, reference_index in enumerate(batch_indices):
                    for layer in registry:
                        activation = outputs["activations"][
                            self._activation_key(layer)
                        ][batch_row]
                        if layer not in edge_layers:
                            for zone in sorted(
                                node_requirements.get(reference_index, ())
                            ):
                                reference_node_values[
                                    (layer, reference_index, zone)
                                ] = activation[zone].clone()
                            continue
                        for source, destination in sorted(
                            edge_requirements.get(reference_index, ())
                        ):
                            matched_edge_index = edge_lookup.get(
                                (source, destination)
                            )
                            if matched_edge_index is None:
                                matched_edge_index = edge_lookup.get(
                                    (source, source)
                                )
                            if matched_edge_index is None:
                                matched_edge_index = next(
                                    (
                                        index
                                        for index, (
                                            ref_source,
                                            _,
                                        ) in enumerate(edges)
                                        if ref_source == source
                                    ),
                                    None,
                                )
                            if matched_edge_index is None:
                                raise RuntimeError(
                                    "matched reference has no feasible "
                                    "source-zone edge"
                                )
                            reference_edge_values[
                                (
                                    layer,
                                    reference_index,
                                    source,
                                    destination,
                                )
                            ] = activation[matched_edge_index].clone()

            node_by_recent: dict[int, list[tuple[int, int]]] = {}
            for recent_index, zone, reference_index in node_pairs:
                node_by_recent.setdefault(recent_index, []).append(
                    (zone, reference_index)
                )
            edge_by_recent: dict[
                int,
                list[tuple[int, int, int, int]],
            ] = {}
            for (
                recent_index,
                edge_index,
                source,
                destination,
                reference_index,
            ) in edge_pairs:
                edge_by_recent.setdefault(recent_index, []).append(
                    (edge_index, source, destination, reference_index)
                )

            selected_recent_indices = sorted(
                set(node_by_recent) | set(edge_by_recent)
            )
            for batch_indices in self._topology_batches(
                recent,
                selected_recent_indices,
            ):
                outputs = self._forward_record_batch(
                    model,
                    [recent[index] for index in batch_indices],
                )
                for batch_row, recent_index in enumerate(batch_indices):
                    for layer in registry:
                        activation = outputs["activations"][
                            self._activation_key(layer)
                        ][batch_row]
                        if layer not in edge_layers:
                            for zone, reference_index in node_by_recent.get(
                                recent_index,
                                (),
                            ):
                                recent_values[layer].append(
                                    activation[zone].clone()
                                )
                                reference_values[layer].append(
                                    reference_node_values[
                                        (layer, reference_index, zone)
                                    ]
                                )
                            continue
                        for (
                            edge_index,
                            source,
                            destination,
                            reference_index,
                        ) in edge_by_recent.get(recent_index, ()):
                            recent_values[layer].append(
                                activation[edge_index].clone()
                            )
                            reference_values[layer].append(
                                reference_edge_values[
                                    (
                                        layer,
                                        reference_index,
                                        source,
                                        destination,
                                    )
                                ]
                            )
        model.train(was_training)
        return (
            {
                layer: torch.stack(values, dim=0)
                for layer, values in recent_values.items()
            },
            {
                layer: torch.stack(values, dim=0)
                for layer, values in reference_values.items()
            },
        )

    def _matched_reference_activation_means(
        self,
        model: MobiWaveDispatchNet,
        recent: Sequence[dict[str, Any]],
        reference: Sequence[dict[str, Any]],
        *,
        match_indices: Sequence[Sequence[int]] | None = None,
        layers: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stream exact matched-reference means without retaining full outputs."""

        full_registry = self._layer_registry(model)
        registry = (
            full_registry
            if layers is None
            else {name: full_registry[name] for name in layers}
        )
        edge_layers = {"return_head", "policy_head"}
        if match_indices is None:
            match_indices = self._reference_record_indices(recent, reference)
        node_counts: dict[int, dict[int, int]] = {}
        edge_counts: dict[int, dict[tuple[int, int], int]] = {}
        for recent_index, record in enumerate(recent):
            for zone in range(self.network.zone_count):
                reference_index = match_indices[recent_index][zone]
                zones = node_counts.setdefault(reference_index, {})
                zones[zone] = zones.get(zone, 0) + 1
            for source, destinations in sorted(
                record["feasible_destinations"].items()
            ):
                for destination in destinations:
                    reference_index = match_indices[recent_index][int(source)]
                    edges = edge_counts.setdefault(reference_index, {})
                    edge = (int(source), int(destination))
                    edges[edge] = edges.get(edge, 0) + 1

        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {name: 0 for name in registry}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            selected_indices = sorted(set(node_counts) | set(edge_counts))
            for batch_indices in self._topology_batches(
                reference,
                selected_indices,
            ):
                outputs = self._forward_record_batch(
                    model,
                    [reference[index] for index in batch_indices],
                )
                model_edges = tuple(
                    (int(source), int(destination))
                    for source, destination in zip(
                        model.edge_sources.detach().cpu().tolist(),
                        model.edge_destinations.detach().cpu().tolist(),
                    )
                )
                edge_lookup = {
                    edge: index for index, edge in enumerate(model_edges)
                }
                for batch_row, reference_index in enumerate(batch_indices):
                    for layer in registry:
                        activation = outputs["activations"][
                            self._activation_key(layer)
                        ][batch_row]
                        layer_sum = sums.get(layer)
                        if layer not in edge_layers:
                            for zone, multiplicity in node_counts.get(
                                reference_index,
                                {},
                            ).items():
                                contribution = (
                                    activation[zone] * multiplicity
                                )
                                layer_sum = (
                                    contribution.clone()
                                    if layer_sum is None
                                    else layer_sum + contribution
                                )
                                counts[layer] += multiplicity
                        else:
                            for (
                                source,
                                destination,
                            ), multiplicity in edge_counts.get(
                                reference_index,
                                {},
                            ).items():
                                matched_index = edge_lookup.get(
                                    (source, destination)
                                )
                                if matched_index is None:
                                    matched_index = edge_lookup.get(
                                        (source, source)
                                    )
                                if matched_index is None:
                                    matched_index = next(
                                        (
                                            index
                                            for index, (
                                                ref_source,
                                                _,
                                            ) in enumerate(model_edges)
                                            if ref_source == source
                                        ),
                                        None,
                                    )
                                if matched_index is None:
                                    raise RuntimeError(
                                        "matched reference has no feasible "
                                        "source-zone edge"
                                    )
                                contribution = (
                                    activation[matched_index] * multiplicity
                                )
                                layer_sum = (
                                    contribution.clone()
                                    if layer_sum is None
                                    else layer_sum + contribution
                                )
                                counts[layer] += multiplicity
                        if layer_sum is not None:
                            sums[layer] = layer_sum
        model.train(was_training)
        if any(counts[name] <= 0 for name in registry):
            raise RuntimeError("matched reference activation mean is empty")
        return {
            name: sums[name] / counts[name]
            for name in registry
        }

    def _matched_spectral_evidence(
        self,
        model: MobiWaveDispatchNet,
        recent: Sequence[dict[str, Any]],
        reference: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current features, stratum-matched reference features, and gates."""

        if not recent or not reference:
            raise ValueError("matched spectral evidence requires nonempty inputs")
        match_indices = self._reference_record_indices(recent, reference)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            recent_scale_features: list[torch.Tensor | None] = [
                None
            ] * len(recent)
            recent_gate_weights: list[torch.Tensor | None] = [
                None
            ] * len(recent)
            for batch_indices in self._topology_batches(
                recent,
                list(range(len(recent))),
            ):
                outputs = self._forward_record_batch(
                    model,
                    [recent[index] for index in batch_indices],
                )
                for batch_row, index in enumerate(batch_indices):
                    recent_scale_features[index] = outputs[
                        "scale_features"
                    ][batch_row].clone()
                    recent_gate_weights[index] = outputs[
                        "gate_weights"
                    ][batch_row].clone()
            if any(value is None for value in recent_scale_features):
                raise RuntimeError("recent spectral feature batch is incomplete")
            if any(value is None for value in recent_gate_weights):
                raise RuntimeError("recent gate-weight batch is incomplete")
            full_recent_features = torch.stack(
                [value for value in recent_scale_features if value is not None],
                dim=0,
            )
            full_recent_gates = torch.stack(
                [value for value in recent_gate_weights if value is not None],
                dim=0,
            )
            flattened_features = full_recent_features.reshape(
                -1,
                full_recent_features.shape[-2],
                full_recent_features.shape[-1],
            )
            sample_count = min(
                int(self.config.dgls_mmd_sample_cap),
                flattened_features.shape[0],
            )
            positions = (
                (
                    (2 * torch.arange(sample_count, device=self.device) + 1)
                    * flattened_features.shape[0]
                )
                // (2 * sample_count)
            ).long()
            recent_features = flattened_features.index_select(0, positions)
            # Preserve the exact full-window dispatch weights while evaluating
            # MMD only on the documented deterministic bounded sample.
            mean_gates = full_recent_gates.reshape(
                -1,
                full_recent_gates.shape[-1],
            ).mean(dim=0)
            recent_gates = mean_gates.unsqueeze(0).expand(sample_count, -1)

            sampled_matches: list[tuple[int, int]] = []
            for position in positions.detach().cpu().tolist():
                recent_index, zone = divmod(
                    int(position),
                    self.network.zone_count,
                )
                sampled_matches.append(
                    (match_indices[recent_index][zone], zone)
                )
            reference_requirements: dict[int, set[int]] = {}
            for index, zone in sampled_matches:
                reference_requirements.setdefault(index, set()).add(zone)
            reference_scale_features: dict[tuple[int, int], torch.Tensor] = {}
            cache_capacity = max(
                1,
                int(self.config.dgls_mmd_sample_cap),
            )
            missing_indices: list[int] = []
            uid_by_index: dict[int, str] = {}
            for index in sorted(reference_requirements):
                record_uid = str(
                    reference[index].get("record_uid", f"index:{index}")
                )
                uid_by_index[index] = record_uid
                cached = self._spectral_reference_cache.get(record_uid)
                if cached is None:
                    missing_indices.append(index)
                else:
                    self._spectral_reference_cache.move_to_end(record_uid)
            for batch_indices in self._topology_batches(
                reference,
                missing_indices,
            ):
                outputs = self._forward_record_batch(
                    model,
                    [reference[index] for index in batch_indices],
                )
                for batch_row, index in enumerate(batch_indices):
                    record_uid = uid_by_index[index]
                    cached = outputs["scale_features"][
                        batch_row
                    ].detach().cpu().clone()
                    self._spectral_reference_cache[record_uid] = cached
                    while len(self._spectral_reference_cache) > cache_capacity:
                        self._spectral_reference_cache.popitem(last=False)
            for index in sorted(reference_requirements):
                cached = self._spectral_reference_cache[uid_by_index[index]]
                for zone in sorted(reference_requirements[index]):
                    reference_scale_features[(index, zone)] = cached[
                        zone
                    ].to(self.device)
            matched_reference = torch.stack(
                [
                    reference_scale_features[(index, zone)]
                    for index, zone in sampled_matches
                ],
                dim=0,
            )
        model.train(was_training)
        return recent_features, matched_reference, recent_gates

    def _reference_match_metadata(
        self,
        reference: Sequence[Mapping[str, Any]],
    ) -> tuple[
        list[int],
        list[np.ndarray],
        dict[tuple[int, int, int], list[int]],
        dict[int, list[int]],
    ]:
        """Build the immutable accepted-reference stratum index once."""

        if (
            self._reference_match_source is reference
            and self._reference_match_source_length == len(reference)
            and self._reference_match_metadata_cache is not None
        ):
            return self._reference_match_metadata_cache

        period = int(self.config.dgls_time_period)
        bins = int(self.config.dgls_time_bins)
        reference_time_bins: list[int] = []
        reference_gaps: list[np.ndarray] = []
        exact_lookup: dict[tuple[int, int, int], list[int]] = {}
        zone_lookup: dict[int, list[int]] = {}
        for index, record in enumerate(reference):
            reference_time_bins.append(
                int((int(record["time"]) % period) * bins / period)
            )
            reference_gaps.append(
                np.asarray(record["backlog"], dtype=float)
                + np.asarray(record["observed_demand"], dtype=float)
                - np.asarray(record["available"], dtype=float)
            )
            active_values = record.get("active_reference_strata")
            if active_values is None:
                active_values = self._reference_strata(record)
            active_zones: set[int] = set()
            for raw_key in active_values:
                key = tuple(int(value) for value in raw_key)
                exact_lookup.setdefault(key, []).append(index)
                active_zones.add(key[1])
            for zone in active_zones:
                zone_lookup.setdefault(zone, []).append(index)

        metadata = (
            reference_time_bins,
            reference_gaps,
            exact_lookup,
            zone_lookup,
        )
        self._reference_match_source = reference
        self._reference_match_source_length = len(reference)
        self._reference_match_metadata_cache = metadata
        return metadata

    def _reference_record_indices(
        self,
        recent: Sequence[Mapping[str, Any]],
        reference: Sequence[Mapping[str, Any]],
    ) -> list[list[int]]:
        """Implement the appendix's deterministic stratum matching.

        Every zone is matched to the same zone in history.  Exact cyclic-time
        and demand--supply-gap bins are preferred; otherwise the closest
        nonempty stratum is selected lexicographically.
        """

        if not reference:
            raise RuntimeError("DGLS reference buffer is empty")
        period = int(self.config.dgls_time_period)
        bins = int(self.config.dgls_time_bins)
        gap_width = float(self.config.dgls_gap_bin_width)

        def time_bin(record: Mapping[str, Any]) -> int:
            return int((int(record["time"]) % period) * bins / period)

        def gap_values(record: Mapping[str, Any]) -> np.ndarray:
            return (
                np.asarray(record["backlog"], dtype=float)
                + np.asarray(record["observed_demand"], dtype=float)
                - np.asarray(record["available"], dtype=float)
            )

        (
            reference_time_bins,
            reference_gaps,
            exact_lookup,
            zone_lookup,
        ) = self._reference_match_metadata(reference)
        matched: list[list[int]] = []
        for current in recent:
            current_time_bin = time_bin(current)
            current_gaps = gap_values(current)
            row: list[int] = []
            for zone in range(self.network.zone_count):
                current_gap_bin = math.floor(current_gaps[zone] / gap_width)
                exact_key = (current_time_bin, zone, current_gap_bin)
                exact = exact_lookup.get(exact_key, ())
                if exact:
                    candidates = exact
                else:
                    candidates = zone_lookup.get(zone, ())
                    if not candidates:
                        raise RuntimeError(
                            "DGLS reference buffer has no active queue for "
                            f"zone {zone}"
                        )

                def fallback_key(index: int) -> tuple[float, float, int]:
                    raw_time_distance = abs(
                        reference_time_bins[index] - current_time_bin
                    )
                    cyclic_time_distance = min(
                        raw_time_distance,
                        bins - raw_time_distance,
                    )
                    standardized_gap_distance = abs(
                        reference_gaps[index][zone] - current_gaps[zone]
                    ) / gap_width
                    return (
                        float(cyclic_time_distance),
                        float(standardized_gap_distance),
                        -int(
                            reference[index].get(
                                "record_sequence",
                                reference[index]["time"],
                            )
                        ),
                    )

                row.append(min(candidates, key=fallback_key))
            matched.append(row)
        return matched

    def _calibrate_bandwidths(
        self,
        model: MobiWaveDispatchNet,
        reference: Sequence[dict[str, Any]],
    ) -> tuple[tuple[float, ...], dict[str, float]]:
        if not reference:
            raise ValueError("bandwidth calibration requires reference records")
        registry = self._layer_registry(model)
        cap = int(self.config.dgls_bandwidth_sample_cap)
        record_count = len(reference)
        node_count = self.network.zone_count
        edge_count = model.edge_count

        def selections(unit_count: int) -> dict[int, list[int]]:
            total = record_count * unit_count
            selected_count = min(cap, total)
            positions = (
                ((2 * np.arange(selected_count) + 1) * total)
                // (2 * selected_count)
            )
            grouped: dict[int, list[int]] = {}
            for position in positions.tolist():
                record_index, unit_index = divmod(int(position), unit_count)
                grouped.setdefault(record_index, []).append(unit_index)
            return grouped

        node_selections = selections(node_count)
        edge_selections = selections(edge_count)
        selected_records = sorted(set(node_selections) | set(edge_selections))
        spectral_rows: list[torch.Tensor] = []
        activation_rows: dict[str, list[torch.Tensor]] = {
            name: [] for name in registry
        }
        edge_layers = {"return_head", "policy_head"}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch_indices in self._topology_batches(
                reference,
                selected_records,
            ):
                outputs = self._forward_record_batch(
                    model,
                    [reference[index] for index in batch_indices],
                )
                for batch_row, record_index in enumerate(batch_indices):
                    node_indices = node_selections.get(record_index, ())
                    if node_indices:
                        index = torch.as_tensor(
                            node_indices,
                            dtype=torch.long,
                            device=self.device,
                        )
                        spectral_rows.append(
                            outputs["scale_features"][
                                batch_row
                            ].index_select(0, index)
                        )
                        for layer in registry:
                            if layer in edge_layers:
                                continue
                            activation = outputs["activations"][
                                self._activation_key(layer)
                            ][batch_row]
                            activation_rows[layer].append(
                                activation.index_select(0, index)
                            )
                    edge_indices = edge_selections.get(record_index, ())
                    if edge_indices:
                        index = torch.as_tensor(
                            edge_indices,
                            dtype=torch.long,
                            device=self.device,
                        )
                        for layer in registry:
                            if layer not in edge_layers:
                                continue
                            activation = outputs["activations"][
                                self._activation_key(layer)
                            ][batch_row]
                            if activation.shape[0] != edge_count:
                                raise RuntimeError(
                                    "offline reference topology changed "
                                    "during fixed-bandwidth calibration"
                                )
                            activation_rows[layer].append(
                                activation.index_select(0, index)
                            )
        model.train(was_training)
        scale_features = torch.cat(spectral_rows, dim=0)
        spectral = tuple(
            self._median_bandwidth(
                scale_features[:, band, :]
            )
            for band in range(scale_features.shape[1])
        )
        activations = {
            name: torch.cat(values, dim=0)
            for name, values in activation_rows.items()
        }
        layer = {
            name: self._median_bandwidth(values)
            for name, values in activations.items()
        }
        return spectral, layer

    def _updated_importance(
        self,
        model: MobiWaveDispatchNet,
        validation: Sequence[dict[str, Any]],
        previous: Mapping[str, Any],
    ) -> dict[str, Any]:
        for parameter in model.parameters():
            parameter.grad = None
        model.eval()
        losses = self._loss_on_records(
            model,
            validation,
            validation,
            include_policy=True,
        )
        losses["total"].backward()
        beta = float(self.config.dgls_importance_beta)
        previous_values = previous["values"]
        values = {}
        for name, parameter in self._adaptable_named_parameters(model):
            gradient = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach()
            )
            values[name] = (
                beta * previous_values[name].to(gradient.device)
                + (1.0 - beta) * gradient.square()
            ).detach().cpu()
        for parameter in model.parameters():
            parameter.grad = None
        return {
            "accepted_count": int(previous["accepted_count"]) + 1,
            "values": values,
        }

    def _effective_layer_rates(
        self,
        selection,
        drift_rate: float,
    ) -> dict[str, float]:
        positive = {name: max(0.0, float(value)) for name, value in selection.scores.items()}
        maximum = max(positive.values(), default=0.0)
        rates = {}
        for layer in selection.selected:
            normalized_score = positive[layer] / max(maximum, 1e-12)
            importance = float(selection.normalized_importance[layer])
            rates[layer] = max(
                1e-12,
                float(drift_rate)
                * normalized_score
                * (
                    float(self.config.dgls_eta_min)
                    + (1.0 - float(self.config.dgls_eta_min))
                    * (1.0 - importance)
                ),
            )
        return rates

    def _new_optimizer(
        self,
        model: MobiWaveDispatchNet,
        registry: Mapping[str, Sequence[str]],
    ) -> DGLSFastSlowOptimizer:
        return DGLSFastSlowOptimizer(
            self._adaptable_named_parameters(model),
            layer_registry=registry,
            lr=float(self.config.dgls_base_lr),
            betas=(
                float(self.config.m3_beta_fast),
                float(self.config.m3_beta_slow),
                float(self.config.m3_beta_second),
            ),
            base_slow_interval=int(self.config.m3_slow_update_interval),
            shock_scale=float(self.config.dgls_kappa_shock),
            persistence_scale=float(self.config.dgls_kappa_persistence),
            interval_min=int(self.config.dgls_slow_interval_min),
            interval_max=int(self.config.dgls_slow_interval_max),
            omega_max=float(self.config.m3_slow_weight),
            newton_schulz_steps=int(self.config.m3_muon_steps),
        )

    def _layer_registry(
        self,
        model: MobiWaveDispatchNet,
    ) -> OrderedDict[str, tuple[str, ...]]:
        names = [
            name
            for name, _ in self._adaptable_named_parameters(model)
        ]

        def under(*prefixes: str) -> tuple[str, ...]:
            return tuple(
                name for name in names if any(name.startswith(prefix) for prefix in prefixes)
            )

        registry: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        registry["input_projection"] = under("graph_wavelet.input_proj.")
        for band in range(model.graph_wavelet.filter_bank.band_count):
            registry[f"scale_encoder_{band}"] = under(
                f"graph_wavelet.scale_encoders.{band}."
            )
        if model.use_gating and model.graph_wavelet.zone_embedding is not None:
            registry["zone_embedding"] = under("graph_wavelet.zone_embedding.")
        if model.use_gating:
            for band in range(model.graph_wavelet.filter_bank.band_count):
                registry[f"gate_{band}"] = under(
                    f"graph_wavelet.gate.gates.{band}."
                )
        registry["residual"] = under("graph_wavelet.residual.")
        registry["demand_head"] = under("demand_hidden.", "demand_output.")
        registry["return_head"] = under("return_hidden.", "return_output.")
        registry["policy_head"] = under("policy_hidden.", "policy_output.")
        registry = OrderedDict(
            (layer, parameter_names)
            for layer, parameter_names in registry.items()
            if parameter_names
        )
        covered = {
            parameter_name
            for parameter_names in registry.values()
            for parameter_name in parameter_names
        }
        if covered != set(names):
            raise RuntimeError(
                "DGLS layer registry does not cover parameters: "
                + ", ".join(sorted(set(names) - covered))
            )
        return registry

    @staticmethod
    def _activation_key(layer: str) -> str:
        return layer

    @staticmethod
    def _adaptable_named_parameters(model: MobiWaveDispatchNet):
        return tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if not (
                not model.use_gating
                and (
                    name.startswith("graph_wavelet.gate.")
                    or name.startswith("graph_wavelet.zone_embedding.")
                )
            )
        )

    @staticmethod
    def _parameter_state(model: MobiWaveDispatchNet) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }

    @staticmethod
    def _load_parameter_state(
        model: MobiWaveDispatchNet,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        state = model._migrate_legacy_parameter_state(state)
        parameters = dict(model.named_parameters())
        if set(parameters) != set(state):
            raise ValueError("MobiWave parameter-state registry mismatch")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = torch.as_tensor(
                    state[name],
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid parameter state for {name}")
                parameter.copy_(value)

    def _importance_hat(
        self,
        importance: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        count = int(importance["accepted_count"])
        if count <= 0:
            return {
                name: torch.zeros_like(value)
                for name, value in importance["values"].items()
            }
        correction = 1.0 - float(self.config.dgls_importance_beta) ** count
        return {
            name: value / max(correction, 1e-12)
            for name, value in importance["values"].items()
        }

    def _demand_target(
        self,
        record: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> np.ndarray | None:
        by_time = {int(item["time"]): item for item in records}
        targets = []
        for offset in range(1, self.model.forecast_horizon + 1):
            target_time = int(record["time"]) + offset
            future = by_time.get(target_time)
            if future is None:
                if int(record.get("terminal_boundary_time", -1)) != target_time:
                    return None
                terminal = record.get("terminal_new_demand")
                if terminal is None:
                    return None
                targets.append(np.asarray(terminal, dtype=np.float32))
            else:
                targets.append(np.asarray(future["new_demand"], dtype=np.float32))
        return np.stack(targets, axis=-1)

    def _materialize_record_labels(
        self,
        record: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Freeze labels before a record crosses a stream/reference boundary."""

        output = dict(record)
        demand_target = self._demand_target(record, records)
        output["resolved_demand_target"] = (
            None
            if demand_target is None
            else np.asarray(demand_target, dtype=np.float32).copy()
        )
        output["resolved_edge_return_targets"] = copy.deepcopy(
            self._edge_return_targets(record, records)
        )
        output["resolved_edge_return_available"] = (
            self._edge_return_window_complete(record, records)
        )
        return output

    def _with_resolved_advantages(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Freeze full-rollout returns before any minibatch optimization."""

        output = [dict(record) for record in records]
        if not output:
            return output
        times = [int(record["time"]) for record in output]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("advantage records must be chronological")
        rewards = np.asarray(
            [float(record["reward"]) for record in output],
            dtype=np.float64,
        )
        returns = np.empty_like(rewards)
        running = 0.0
        gamma = float(self.config.policy_discount)
        for index in range(len(output) - 1, -1, -1):
            running = float(rewards[index]) + gamma * running
            returns[index] = running
        centered = returns - float(returns.mean())
        scale = float(np.sqrt(np.mean(centered**2))) + 1e-8
        for record, advantage in zip(output, centered / scale):
            record["resolved_advantage"] = float(advantage)
        return output

    def _compact_reference_records(
        self,
        records: Sequence[Mapping[str, Any]],
        capacity: int,
    ) -> list[dict[str, Any]]:
        """Materialize fixed-capacity FIFO queues for every reference stratum.

        A compact full-graph record may support one queue per zone.  The record
        itself is stored once, while ``active_reference_strata`` records the
        queues in which it remains live.  This preserves the appendix's
        per-stratum FIFO semantics without duplicating a full graph tensor for
        every zone.
        """

        if capacity <= 0:
            raise ValueError("reference capacity must be positive")
        # A sliding validation window can expose the same historical step more
        # than once.  Keep its first accepted copy so it is not re-appended to
        # queues from which it has already been evicted.
        unique: dict[str, dict[str, Any]] = {}
        for ordinal, raw_record in enumerate(records):
            value = dict(raw_record)
            identity = str(value.get("record_uid", "")).strip()
            if not identity:
                # Backward-compatible identity for checkpoints written before
                # stream-aware IDs.  The generated value is persisted by the
                # compact record, so subsequent compactions remain stable.
                identity = (
                    f"legacy:{ordinal}:time={int(value['time'])}:"
                    f"graph={str(value.get('graph_version', ''))}"
                )
                value["record_uid"] = identity
                value.setdefault("stream_id", "legacy")
            value.setdefault("record_sequence", ordinal + 1)
            existing = unique.get(identity)
            if existing is None:
                unique[identity] = value
                continue
            # A record can first enter an accepted reference suffix before its
            # H-step targets have matured.  A later accepted window enriches
            # its labels without changing FIFO identity or queue position.
            if (
                existing.get("resolved_demand_target") is None
                and value.get("resolved_demand_target") is not None
            ):
                existing["resolved_demand_target"] = value[
                    "resolved_demand_target"
                ]
            if (
                not bool(existing.get("resolved_edge_return_available", False))
                and bool(value.get("resolved_edge_return_available", False))
            ):
                existing["resolved_edge_return_targets"] = value.get(
                    "resolved_edge_return_targets",
                    {},
                )
                existing["resolved_edge_return_available"] = True
        ordered = list(unique.values())
        ordered.sort(
            key=lambda record: int(
                record.get("record_sequence", record["time"])
            )
        )
        if not ordered:
            return []

        queues: dict[tuple[int, int, int], list[int]] = {}
        for index, record in enumerate(ordered):
            active = record.get("active_reference_strata")
            strata = (
                tuple(
                    tuple(int(value) for value in key)
                    for key in active
                )
                if active is not None
                else self._reference_strata(record)
            )
            for key in strata:
                queue = queues.setdefault(key, [])
                queue.append(index)
                if len(queue) > capacity:
                    del queue[: len(queue) - capacity]

        active_by_record: dict[int, list[tuple[int, int, int]]] = {}
        for key, queue in queues.items():
            for index in queue:
                active_by_record.setdefault(index, []).append(key)

        compact: list[dict[str, Any]] = []
        for index in sorted(
            active_by_record,
            key=lambda value: int(
                ordered[value].get(
                    "record_sequence",
                    ordered[value]["time"],
                )
            ),
        ):
            record = self._compact_reference_record(ordered[index])
            record["active_reference_strata"] = tuple(
                sorted(active_by_record[index])
            )
            compact.append(record)
        return compact

    def _reference_strata(
        self,
        record: Mapping[str, Any],
    ) -> tuple[tuple[int, int, int], ...]:
        period = int(self.config.dgls_time_period)
        bins = int(self.config.dgls_time_bins)
        gap_width = float(self.config.dgls_gap_bin_width)
        time_bin = int((int(record["time"]) % period) * bins / period)
        gaps = (
            np.asarray(record["backlog"], dtype=float)
            + np.asarray(record["observed_demand"], dtype=float)
            - np.asarray(record["available"], dtype=float)
        )
        return tuple(
            (
                time_bin,
                zone,
                math.floor(float(gaps[zone]) / gap_width),
            )
            for zone in range(self.network.zone_count)
        )

    @staticmethod
    def _compact_reference_record(
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strip rollout-only fields and quantize large immutable inputs."""

        required = (
            "time",
            "features",
            "available",
            "backlog",
            "observed_demand",
            "external_context",
            "feasible_destinations",
            "edge_costs",
            "edge_graph_weights",
            "graph_version",
        )
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(
                "reference record is missing fields: " + ", ".join(missing)
            )
        if (
            "travel_signature" not in record
            and "edge_travel_times" not in record
        ):
            raise ValueError(
                "reference record is missing travel-time evidence"
            )
        topology_signature = record.get("topology_signature")
        if topology_signature is None:
            feasible = record["feasible_destinations"]
            weights = record["edge_graph_weights"]
            topology_signature = tuple(
                (
                    int(source),
                    int(destination),
                    float(
                        0.0
                        if int(source) == int(destination)
                        else weights.get(
                            (int(source), int(destination)),
                            1.0,
                        )
                    ),
                )
                for source, destinations in sorted(feasible.items())
                for destination in sorted(destinations)
            )
        travel_signature = record.get("travel_signature")
        if travel_signature is None:
            travel_lookup = record["edge_travel_times"]
            travel_signature = tuple(
                (
                    int(source),
                    int(destination),
                    float(
                        travel_lookup[(int(source), int(destination))]
                    ),
                )
                for source, destination, _ in topology_signature
            )
        return {
            "time": int(record["time"]),
            "stream_id": str(record.get("stream_id", "legacy")),
            "record_uid": str(record["record_uid"]),
            "record_sequence": int(
                record.get("record_sequence", record["time"])
            ),
            "features": np.asarray(
                record["features"],
                dtype=np.float16,
            ).copy(),
            "available": np.asarray(
                record["available"],
                dtype=np.int16,
            ).copy(),
            "backlog": np.asarray(
                record["backlog"],
                dtype=np.float16,
            ).copy(),
            "observed_demand": np.asarray(
                record["observed_demand"],
                dtype=np.float16,
            ).copy(),
            "external_context": np.asarray(
                record["external_context"],
                dtype=np.float16,
            ).copy(),
            "feasible_destinations": record["feasible_destinations"],
            "edge_costs": record["edge_costs"],
            "edge_graph_weights": record["edge_graph_weights"],
            "graph_version": str(record["graph_version"]),
            "topology_signature": topology_signature,
            "travel_signature": travel_signature,
            "resolved_demand_target": (
                None
                if record.get("resolved_demand_target") is None
                else np.asarray(
                    record["resolved_demand_target"],
                    dtype=np.float32,
                ).copy()
            ),
            "resolved_edge_return_targets": copy.deepcopy(
                record.get("resolved_edge_return_targets", {})
            ),
            "resolved_edge_return_available": bool(
                record.get("resolved_edge_return_available", False)
            ),
        }

    def _edge_return_targets(
        self,
        record: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[int, int], float]:
        """Aggregate each executed move's observed H-step per-vehicle net return."""

        if not self._edge_return_window_complete(record, records):
            return {}
        edge_vehicle_ids = record.get("edge_vehicle_ids")
        if not edge_vehicle_ids:
            return {
                tuple(edge): float(value)
                for edge, value in record.get("edge_return_targets", {}).items()
            }
        by_time = {int(item["time"]): item for item in records}
        start_time = int(record["time"])
        gamma = float(self.config.policy_discount)
        vehicles_by_destination: dict[int, int] = {}
        for raw_edge, vehicle_ids in edge_vehicle_ids.items():
            destination = int(tuple(raw_edge)[1])
            vehicles_by_destination[destination] = (
                vehicles_by_destination.get(destination, 0)
                + len(vehicle_ids)
            )
        targets: dict[tuple[int, int], float] = {}
        for raw_edge, vehicle_ids in edge_vehicle_ids.items():
            edge = tuple(raw_edge)
            per_vehicle = []
            for vehicle_id in vehicle_ids:
                net_return = 0.0
                observed = False
                for offset in range(self.model.forecast_horizon):
                    future = by_time[start_time + offset]
                    rewards = future.get("vehicle_rewards", {})
                    if int(vehicle_id) in rewards:
                        net_return += (gamma**offset) * float(rewards[int(vehicle_id)])
                        observed = True
                if observed:
                    per_vehicle.append(net_return)
            cancellation_share = 0.0
            destination = int(edge[1])
            destination_vehicle_count = max(
                1,
                vehicles_by_destination.get(destination, 0),
            )
            # Requests canceled at t expired before U_t and cannot be caused
            # by it.  Only later decision boundaries are attributed here.
            for offset in range(1, self.model.forecast_horizon + 1):
                boundary_time = start_time + offset
                future = by_time.get(boundary_time)
                if future is not None:
                    canceled = future.get("canceled_by_zone", {})
                else:
                    canceled = record.get("terminal_canceled_by_zone", {})
                cancellation_share += (
                    (gamma ** (offset - 1))
                    * float(canceled.get(destination, 0))
                    * float(self.config.cancel_penalty)
                    / destination_vehicle_count
                )
            if per_vehicle or cancellation_share:
                targets[edge] = (
                    float(np.mean(per_vehicle)) if per_vehicle else 0.0
                ) - cancellation_share
        return targets

    def _edge_return_window_complete(
        self,
        record: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Return whether every reward and terminal boundary in H is observed."""

        by_time = {int(item["time"]): item for item in records}
        start_time = int(record["time"])
        horizon = int(self.model.forecast_horizon)
        if any(
            start_time + offset not in by_time
            for offset in range(horizon)
        ):
            return False
        boundary_time = start_time + horizon
        return (
            boundary_time in by_time
            or (
                int(record.get("terminal_boundary_time", -1))
                == boundary_time
                and "terminal_canceled_by_zone" in record
            )
        )

    def _credit_predecision_cancellations(
        self,
        snapshot: Mapping[str, Any],
    ) -> None:
        """Attach cancellations observed at t to the preceding action reward."""

        if not self._records:
            return
        time = int(snapshot["time"])
        previous = self._records[-1]
        if int(previous["time"]) != time - 1:
            return
        if int(previous.get("credited_cancellation_time", -1)) == time:
            return
        canceled = int(snapshot.get("canceled_before", 0))
        if canceled:
            previous["reward"] = float(previous["reward"]) - (
                float(self.config.cancel_penalty) * canceled
            )
        previous["credited_cancellation_time"] = time

    def _median_bandwidth(self, values: torch.Tensor) -> float:
        floor = float(self.config.dgls_bandwidth_floor)
        flattened = values.detach().reshape(values.shape[0], -1).float()
        flattened = deterministic_row_sample(
            flattened,
            int(self.config.dgls_bandwidth_sample_cap),
        )
        if flattened.shape[0] < 2:
            return floor
        distances = torch.pdist(flattened)
        positive = distances[distances > 0]
        if positive.numel() == 0:
            return floor
        return max(floor, float(positive.median().cpu().item()))

    @staticmethod
    def _request_signature(requests: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            sorted(
                (
                    int(request.request_id),
                    int(request.origin),
                    int(request.destination),
                    int(request.created_at),
                    float(request.fare),
                )
                for request in requests
            )
        )

    @contextmanager
    def _isolated_torch_rng(self):
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        mps_state = None
        if self.device.type == "mps" and hasattr(torch, "mps"):
            getter = getattr(torch.mps, "get_rng_state", None)
            if getter is not None:
                mps_state = getter()
        with torch.random.fork_rng(devices=cuda_devices):
            try:
                yield
            finally:
                if mps_state is not None:
                    setter = getattr(torch.mps, "set_rng_state", None)
                    if setter is not None:
                        setter(mps_state)


def registry_parameters(
    registry: Mapping[str, Sequence[str]],
    selected_layers: Sequence[str],
) -> tuple[str, ...]:
    selected = set(selected_layers)
    return tuple(
        parameter_name
        for layer, parameter_names in registry.items()
        if layer in selected
        for parameter_name in parameter_names
    )


__all__ = ["DGLSMobiWavePolicy", "REQUIRED_VIOLATIONS"]
