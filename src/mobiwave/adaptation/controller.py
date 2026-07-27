from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import math

import torch

from .dgls import CandidateMetrics, LayerSelection, rbf_mmd2, select_layers_under_budget
from .state import (
    AcceptedState,
    CandidateState,
    CausalSplit,
    LayerSignals,
    MutableState,
)


EVENT_SCHEMA = (
    "event_id",
    "event_type",
    "decision_time",
    "transaction_id",
    "base_version",
    "current_version",
    "drift_score",
    "band_scores",
    "band_weights",
    "S",
    "P",
    "active",
    "selected_layers",
    "selected_cost",
    "budget",
    "eta",
    "accepted",
    "reward_gain",
    "violation_deltas",
    "reason",
)

REPORT_SCHEMA = (
    "accepted_version",
    "drift_state",
    "pending_transaction_ids",
    "finalizing_transaction_ids",
    "finalized_transaction_ids",
    "events",
)


@dataclass(frozen=True)
class ControllerConfig:
    bandwidths: tuple[float, ...]
    threshold_on: float
    threshold_off: float
    budget_ratio: float
    mmd_sample_cap: int = 256
    persistence_decay: float = 0.9
    eta_base: float = 1e-3
    eta_min: float = 1e-5
    eta_max: float = 1e-2
    eta_shock_scale: float = 1.0
    eta_persistence_scale: float = 1.0
    weight_activation: float = 0.45
    weight_gradient: float = 0.45
    weight_importance: float = 0.10
    reward_margin: float = 0.0
    required_violations: tuple[str, ...] = ()
    violation_tolerances: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if not bandwidths or any(not math.isfinite(value) or value <= 0.0 for value in bandwidths):
            raise ValueError("bandwidths must contain fixed finite positive values")
        numeric_nonnegative = (
            self.eta_base,
            self.eta_min,
            self.eta_max,
            self.eta_shock_scale,
            self.eta_persistence_scale,
            self.weight_activation,
            self.weight_gradient,
            self.weight_importance,
            self.reward_margin,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numeric_nonnegative):
            raise ValueError("Controller scales, weights, eta values, and reward margin must be nonnegative")
        if not (self.threshold_on > self.threshold_off >= 0.0):
            raise ValueError("threshold_on must exceed nonnegative threshold_off")
        if not 0.0 < float(self.budget_ratio) <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")
        if (
            isinstance(self.mmd_sample_cap, bool)
            or not isinstance(self.mmd_sample_cap, int)
            or self.mmd_sample_cap < 2
        ):
            raise ValueError("mmd_sample_cap must be an integer of at least two")
        if not 0.0 <= float(self.persistence_decay) < 1.0:
            raise ValueError("persistence_decay must be in [0, 1)")
        if not (0.0 < float(self.eta_min) <= float(self.eta_base) <= float(self.eta_max)):
            raise ValueError("eta values must satisfy 0 < eta_min <= eta_base <= eta_max")
        required = tuple(str(name).strip() for name in self.required_violations)
        if any(not name for name in required) or len(set(required)) != len(required):
            raise ValueError("required_violations must contain unique nonempty names")
        tolerances = {str(name): float(value) for name, value in self.violation_tolerances.items()}
        if any(not name for name in tolerances):
            raise ValueError("violation tolerance names must be nonempty")
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances.values()):
            raise ValueError("violation tolerances must be finite and nonnegative")
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "required_violations", required)
        object.__setattr__(self, "violation_tolerances", MappingProxyType(tolerances))


@dataclass(frozen=True)
class SpectralDrift:
    score: float
    band_scores: tuple[float, ...]
    band_weights: tuple[float, ...]


def fixed_band_spectral_drift(
    recent_features: torch.Tensor,
    reference_features: torch.Tensor,
    recent_gate_weights: torch.Tensor,
    bandwidths: Sequence[float],
    sample_cap: int | None = None,
) -> SpectralDrift:
    """Dispatch-weighted spectral drift with one immutable bandwidth per band."""

    if recent_features.ndim < 3 or reference_features.ndim < 3:
        raise ValueError("Spectral features must include sample, band, and feature dimensions")
    if recent_features.shape[-2:] != reference_features.shape[-2:]:
        raise ValueError("Recent and reference band/feature dimensions must match")
    band_count = int(recent_features.shape[-2])
    checked_bandwidths = tuple(float(value) for value in bandwidths)
    if len(checked_bandwidths) != band_count:
        raise ValueError("Exactly one fixed bandwidth is required per spectral band")
    if any(not math.isfinite(value) or value <= 0.0 for value in checked_bandwidths):
        raise ValueError("Every fixed spectral bandwidth must be finite and positive")
    if tuple(recent_gate_weights.shape) != tuple(recent_features.shape[:-1]):
        raise ValueError(
            "Gate weights must match every recent sample/node and spectral band"
        )
    if not torch.isfinite(recent_features).all() or not torch.isfinite(reference_features).all():
        raise ValueError("Spectral features must be finite")
    gates = recent_gate_weights.reshape(-1, band_count).float()
    if not torch.isfinite(gates).all() or bool((gates < 0.0).any()):
        raise ValueError("Gate weights must be finite and nonnegative")
    mean_weights = gates.mean(dim=0)
    total_weight = mean_weights.sum()
    if float(total_weight.detach().cpu().item()) <= 0.0:
        raise ValueError("Gate weights must have positive total mass")
    mean_weights = mean_weights / total_weight
    width = int(recent_features.shape[-1])
    recent = recent_features.reshape(-1, band_count, width)
    reference = reference_features.reshape(-1, band_count, width)
    band_values = torch.stack(
        [
            rbf_mmd2(
                recent[:, band],
                reference[:, band],
                checked_bandwidths[band],
                sample_cap=sample_cap,
            )
            for band in range(band_count)
        ]
    )
    score = torch.sum(mean_weights.to(device=band_values.device) * band_values)
    return SpectralDrift(
        score=float(score.detach().cpu().item()),
        band_scores=tuple(float(value) for value in band_values.detach().cpu().tolist()),
        band_weights=tuple(float(value) for value in mean_weights.detach().cpu().tolist()),
    )


@dataclass(frozen=True)
class DriftDynamics:
    S: float = 0.0
    P: float = 0.0
    previous_score: float = 0.0
    active: bool = False


@dataclass(frozen=True)
class AdaptationContext:
    transaction_id: str
    decision_time: int
    base_version: int
    drift: SpectralDrift
    dynamics: DriftDynamics
    selection: LayerSelection
    eta: float


@dataclass(frozen=True)
class CandidateTransaction:
    context: AdaptationContext


@dataclass(frozen=True)
class _PendingTransaction:
    handle: CandidateTransaction
    context: AdaptationContext
    base_state: AcceptedState
    split: CausalSplit
    candidate: CandidateState


@dataclass(frozen=True)
class ValidationPair:
    stable: CandidateMetrics
    candidate: CandidateMetrics


@dataclass(frozen=True)
class TransactionResult:
    accepted: bool
    reason: str
    state: AcceptedState
    event: Mapping[str, Any]


@dataclass(frozen=True)
class DGLSEvent:
    event_id: int
    event_type: str
    decision_time: int
    transaction_id: str
    base_version: int
    current_version: int
    drift_score: float
    band_scores: tuple[float, ...]
    band_weights: tuple[float, ...]
    S: float
    P: float
    active: bool
    selected_layers: tuple[str, ...]
    selected_cost: float
    budget: float
    eta: float
    accepted: bool | None
    reward_gain: float | None
    violation_deltas: Mapping[str, float]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "decision_time": self.decision_time,
            "transaction_id": self.transaction_id,
            "base_version": self.base_version,
            "current_version": self.current_version,
            "drift_score": self.drift_score,
            "band_scores": list(self.band_scores),
            "band_weights": list(self.band_weights),
            "S": self.S,
            "P": self.P,
            "active": self.active,
            "selected_layers": list(self.selected_layers),
            "selected_cost": self.selected_cost,
            "budget": self.budget,
            "eta": self.eta,
            "accepted": self.accepted,
            "reward_gain": self.reward_gain,
            "violation_deltas": dict(self.violation_deltas),
            "reason": self.reason,
        }
        if tuple(payload) != EVENT_SCHEMA:
            raise RuntimeError("DGLS event schema changed unexpectedly")
        return payload


CandidateCallback = Callable[[MutableState, CausalSplit, AdaptationContext], CandidateState | MutableState]
LayerSignalsSource = LayerSignals | Callable[[], LayerSignals]
ValidationCallback = Callable[[AcceptedState, CandidateState, tuple[Any, ...]], ValidationPair]


def copy_event_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(event),
            "band_scores": list(event["band_scores"]),
            "band_weights": list(event["band_weights"]),
            "selected_layers": list(event["selected_layers"]),
            "violation_deltas": dict(event["violation_deltas"]),
        }
        for event in events
    ]


class DGLSTransactionController:
    """Model-agnostic DGLS transaction coordinator with CAS commits."""

    def __init__(self, accepted: AcceptedState, config: ControllerConfig) -> None:
        self.config = config
        self._accepted = accepted.clone()
        self._dynamics = DriftDynamics()
        self._lock = RLock()
        self._transaction_counter = 0
        self._event_counter = 0
        self._events: list[dict[str, Any]] = []
        self._pending: dict[str, _PendingTransaction] = {}
        self._finalizing: set[str] = set()
        self._finalized: set[str] = set()

    @property
    def accepted(self) -> AcceptedState:
        with self._lock:
            return self._accepted.clone()

    def materialize_accepted(self) -> MutableState:
        """Return one isolated mutable copy without an intermediate snapshot."""

        with self._lock:
            return self._accepted.materialize()

    @property
    def dynamics(self) -> DriftDynamics:
        with self._lock:
            return DriftDynamics(
                S=self._dynamics.S,
                P=self._dynamics.P,
                previous_score=self._dynamics.previous_score,
                active=self._dynamics.active,
            )

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                {
                    **event,
                    "band_scores": list(event["band_scores"]),
                    "band_weights": list(event["band_weights"]),
                    "selected_layers": list(event["selected_layers"]),
                    "violation_deltas": dict(event["violation_deltas"]),
                }
                for event in self._events
            )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot accepted and stream state when no transaction is in flight."""

        with self._lock:
            if self._pending or self._finalizing:
                raise RuntimeError("Cannot checkpoint DGLS with a transaction in flight")
            return {
                "accepted": {
                    "model_state": self._accepted.model_state,
                    "optimizer_state": self._accepted.optimizer_state,
                    "importance_state": self._accepted.importance_state,
                    "reference_state": self._accepted.reference_state,
                    "version": self._accepted.version,
                },
                "dynamics": {
                    "S": self._dynamics.S,
                    "P": self._dynamics.P,
                    "previous_score": self._dynamics.previous_score,
                    "active": self._dynamics.active,
                },
                "transaction_counter": self._transaction_counter,
                "event_counter": self._event_counter,
                "events": copy_event_rows(self._events),
                "finalized": tuple(sorted(self._finalized)),
            }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Validate a complete checkpoint before atomically replacing state."""

        source = dict(payload)
        accepted_payload = dict(source["accepted"])
        accepted = AcceptedState(
            model_state=accepted_payload["model_state"],
            optimizer_state=accepted_payload["optimizer_state"],
            importance_state=accepted_payload["importance_state"],
            reference_state=accepted_payload["reference_state"],
            version=int(accepted_payload["version"]),
        )
        dynamics_payload = dict(source["dynamics"])
        dynamics = DriftDynamics(
            S=float(dynamics_payload["S"]),
            P=float(dynamics_payload["P"]),
            previous_score=float(dynamics_payload["previous_score"]),
            active=bool(dynamics_payload["active"]),
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (dynamics.S, dynamics.P, dynamics.previous_score)
        ):
            raise ValueError("Invalid DGLS drift dynamics checkpoint")
        transaction_counter = int(source["transaction_counter"])
        event_counter = int(source["event_counter"])
        if transaction_counter < 0 or event_counter < 0:
            raise ValueError("Invalid DGLS checkpoint counters")
        events = copy_event_rows(source.get("events", ()))
        if any(tuple(event) != EVENT_SCHEMA for event in events):
            raise ValueError("DGLS event checkpoint has an incompatible schema")
        finalized = {str(value) for value in source.get("finalized", ())}
        with self._lock:
            if self._pending or self._finalizing:
                raise RuntimeError("Cannot restore DGLS with a transaction in flight")
            self._accepted = accepted
            self._dynamics = dynamics
            self._transaction_counter = transaction_counter
            self._event_counter = event_counter
            self._events = events
            self._finalized = finalized
            self._pending = {}

    def report(self) -> dict[str, Any]:
        """Return a detached, JSON-ready view of controller state and events."""

        with self._lock:
            payload = {
                "accepted_version": self._accepted.version,
                "drift_state": {
                    "S": self._dynamics.S,
                    "P": self._dynamics.P,
                    "previous_score": self._dynamics.previous_score,
                    "active": self._dynamics.active,
                },
                "pending_transaction_ids": sorted(self._pending),
                "finalizing_transaction_ids": sorted(self._finalizing),
                "finalized_transaction_ids": sorted(self._finalized),
                "events": list(self.events),
            }
            if tuple(payload) != REPORT_SCHEMA:
                raise RuntimeError("DGLS report schema changed unexpectedly")
            return payload

    def propose(
        self,
        *,
        split: CausalSplit,
        recent_features: torch.Tensor,
        reference_features: torch.Tensor,
        recent_gate_weights: torch.Tensor,
        layer_signals: LayerSignalsSource,
        candidate_callback: CandidateCallback,
    ) -> CandidateTransaction | None:
        private_split = split.clone()
        if private_split.decision_time < 0:
            raise ValueError("CausalSplit decision_time must be nonnegative")
        drift = fixed_band_spectral_drift(
            recent_features,
            reference_features,
            recent_gate_weights,
            self.config.bandwidths,
            self.config.mmd_sample_cap,
        )
        with self._lock:
            self._dynamics = self._next_dynamics(drift.score)
            dynamics = self._dynamics
            base_version = self._accepted.version
            self._append_event(
                event_type="drift_assessed",
                decision_time=private_split.decision_time,
                transaction_id="",
                base_version=base_version,
                drift=drift,
                dynamics=dynamics,
                selection=None,
                eta=0.0,
                accepted=None,
                reward_gain=None,
                violation_deltas={},
                reason="active" if dynamics.active else "below_trigger",
            )
            accepted = self._accepted.clone() if dynamics.active else None
        if not dynamics.active:
            return None
        if accepted is None:  # Defensive narrowing for static type checkers.
            raise RuntimeError("active DGLS state was not materialized")

        resolved_layer_signals = (
            layer_signals() if callable(layer_signals) else layer_signals
        )
        if not isinstance(resolved_layer_signals, LayerSignals):
            raise TypeError("layer_signals callback must return LayerSignals")
        selection = select_layers_under_budget(
            resolved_layer_signals.activation,
            resolved_layer_signals.gradient,
            resolved_layer_signals.importance,
            resolved_layer_signals.costs,
            self.config.budget_ratio,
            weight_activation=self.config.weight_activation,
            weight_gradient=self.config.weight_gradient,
            weight_importance=self.config.weight_importance,
        )
        eta = self._eta(dynamics)
        with self._lock:
            self._transaction_counter += 1
            transaction_id = f"tx-{self._transaction_counter:08d}"
        context = AdaptationContext(
            transaction_id=transaction_id,
            decision_time=private_split.decision_time,
            base_version=accepted.version,
            drift=drift,
            dynamics=dynamics,
            selection=selection,
            eta=eta,
        )
        if not selection.selected:
            with self._lock:
                self._append_event(
                    event_type="candidate_rolled_back",
                    decision_time=private_split.decision_time,
                    transaction_id=transaction_id,
                    base_version=accepted.version,
                    drift=drift,
                    dynamics=dynamics,
                    selection=selection,
                    eta=eta,
                    accepted=False,
                    reward_gain=None,
                    violation_deltas={},
                    reason="no_layer_within_budget",
                )
            return None

        materialized = accepted.materialize()
        try:
            produced = candidate_callback(
                materialized,
                private_split.clone(),
                context,
            )
            if isinstance(produced, MutableState):
                candidate = CandidateState.from_mutable(produced)
            elif isinstance(produced, CandidateState):
                candidate = produced.clone()
            else:
                raise TypeError("candidate_callback must return CandidateState or MutableState")
        except Exception as exc:
            with self._lock:
                self._append_event(
                    event_type="candidate_rolled_back",
                    decision_time=private_split.decision_time,
                    transaction_id=transaction_id,
                    base_version=accepted.version,
                    drift=drift,
                    dynamics=dynamics,
                    selection=selection,
                    eta=eta,
                    accepted=False,
                    reward_gain=None,
                    violation_deltas={},
                    reason=f"candidate_callback:{type(exc).__name__}",
                )
            return None

        transaction = CandidateTransaction(context=context)
        pending = _PendingTransaction(
            handle=transaction,
            context=context,
            base_state=accepted,
            split=private_split,
            candidate=candidate,
        )
        with self._lock:
            self._append_event(
                event_type="candidate_proposed",
                decision_time=private_split.decision_time,
                transaction_id=transaction_id,
                base_version=accepted.version,
                drift=drift,
                dynamics=dynamics,
                selection=selection,
                eta=eta,
                accepted=None,
                reward_gain=None,
                violation_deltas={},
                reason="candidate_isolated",
            )
            self._pending[transaction_id] = pending
        return transaction

    def finalize(
        self,
        transaction: CandidateTransaction,
        validation_callback: ValidationCallback,
    ) -> TransactionResult:
        transaction_id = transaction.context.transaction_id
        with self._lock:
            if transaction_id in self._finalized or transaction_id in self._finalizing:
                raise RuntimeError(f"Transaction {transaction_id} has already been finalized")
            pending = self._pending.get(transaction_id)
            if pending is None:
                raise RuntimeError(f"Transaction {transaction_id} was not issued by this controller")
            if pending.handle is not transaction:
                raise RuntimeError(f"Transaction {transaction_id} payload does not match the issued transaction")
            self._finalizing.add(transaction_id)

        try:
            pair = validation_callback(
                pending.base_state.clone(),
                pending.candidate.clone(),
                pending.split.validation,
            )
            if not isinstance(pair, ValidationPair):
                raise TypeError("validation_callback must return ValidationPair")
            valid, reward_gain, deltas, reason = self._validation_decision(pair)
        except Exception as exc:
            valid = False
            reward_gain = None
            deltas = {}
            reason = f"validation_callback:{type(exc).__name__}: {exc}"

        with self._lock:
            try:
                current = self._accepted
                if current.version != pending.context.base_version:
                    valid = False
                    reason = "stale_base_version"
                if valid:
                    try:
                        next_state = AcceptedState.from_candidate(
                            pending.candidate,
                            version=current.version + 1,
                        )
                    except Exception as exc:
                        valid = False
                        reason = f"commit:{type(exc).__name__}"
                        next_state = current
                        event_type = "candidate_rolled_back"
                    else:
                        self._accepted = next_state
                        event_type = "candidate_committed"
                else:
                    next_state = current
                    event_type = "candidate_rolled_back"
                event = self._append_event(
                    event_type=event_type,
                    decision_time=pending.context.decision_time,
                    transaction_id=transaction_id,
                    base_version=pending.context.base_version,
                    drift=pending.context.drift,
                    dynamics=pending.context.dynamics,
                    selection=pending.context.selection,
                    eta=pending.context.eta,
                    accepted=bool(valid),
                    reward_gain=reward_gain,
                    violation_deltas=deltas,
                    reason=reason,
                )
                self._finalized.add(transaction_id)
                return TransactionResult(
                    accepted=bool(valid),
                    reason=reason,
                    state=next_state.clone(),
                    event=event,
                )
            finally:
                self._finalizing.discard(transaction_id)
                self._pending.pop(transaction_id, None)

    def _next_dynamics(self, score: float) -> DriftDynamics:
        previous = self._dynamics
        shock = max(0.0, float(score) - previous.previous_score)
        persistence = (
            self.config.persistence_decay * previous.P
            + (1.0 - self.config.persistence_decay) * float(score)
        )
        if score >= self.config.threshold_on:
            active = True
        elif score <= self.config.threshold_off:
            active = False
        else:
            active = previous.active
        return DriftDynamics(
            S=shock,
            P=persistence,
            previous_score=float(score),
            active=active,
        )

    def _eta(self, dynamics: DriftDynamics) -> float:
        numerator = self.config.eta_base * (
            1.0 + self.config.eta_shock_scale * dynamics.S
        )
        denominator = 1.0 + self.config.eta_persistence_scale * dynamics.P
        return min(self.config.eta_max, max(self.config.eta_min, numerator / denominator))

    def _validation_decision(
        self,
        pair: ValidationPair,
    ) -> tuple[bool, float, dict[str, float], str]:
        stable = pair.stable
        candidate = pair.candidate
        if not math.isfinite(float(stable.reward)) or not math.isfinite(float(candidate.reward)):
            raise ValueError("Validation rewards must be finite")
        required = set(self.config.required_violations)
        missing_stable = required - set(stable.violations)
        missing_candidate = required - set(candidate.violations)
        if missing_stable or missing_candidate:
            missing = sorted(missing_stable | missing_candidate)
            return False, float(candidate.reward) - float(stable.reward), {}, (
                "missing_required_violations:" + ",".join(missing)
            )
        names = tuple(sorted(set(stable.violations) | set(candidate.violations)))
        deltas: dict[str, float] = {}
        failed = []
        for name in names:
            stable_value = float(stable.violations.get(name, 0.0))
            candidate_value = float(candidate.violations.get(name, 0.0))
            if not math.isfinite(stable_value) or not math.isfinite(candidate_value):
                raise ValueError("Validation violations must be finite")
            tolerance = float(self.config.violation_tolerances.get(name, 0.0))
            delta = candidate_value - stable_value
            deltas[name] = delta
            if delta > tolerance:
                failed.append(name)
        reward_gain = float(candidate.reward) - float(stable.reward)
        # Strict improvement is intentional: equality is not an improvement.
        reward_passed = (
            reward_gain > 0.0
            and reward_gain >= self.config.reward_margin
        )
        if not reward_passed:
            return False, reward_gain, deltas, "reward_not_improved"
        if failed:
            return False, reward_gain, deltas, "constraint:" + ",".join(failed)
        return True, reward_gain, deltas, "accepted"

    def _append_event(
        self,
        *,
        event_type: str,
        decision_time: int,
        transaction_id: str,
        base_version: int,
        drift: SpectralDrift,
        dynamics: DriftDynamics,
        selection: LayerSelection | None,
        eta: float,
        accepted: bool | None,
        reward_gain: float | None,
        violation_deltas: Mapping[str, float],
        reason: str,
    ) -> dict[str, Any]:
        self._event_counter += 1
        event = DGLSEvent(
            event_id=self._event_counter,
            event_type=event_type,
            decision_time=int(decision_time),
            transaction_id=transaction_id,
            base_version=int(base_version),
            current_version=self._accepted.version,
            drift_score=drift.score,
            band_scores=drift.band_scores,
            band_weights=drift.band_weights,
            S=dynamics.S,
            P=dynamics.P,
            active=dynamics.active,
            selected_layers=selection.selected if selection is not None else (),
            selected_cost=selection.selected_cost if selection is not None else 0.0,
            budget=selection.budget if selection is not None else 0.0,
            eta=float(eta),
            accepted=accepted,
            reward_gain=reward_gain,
            violation_deltas=dict(violation_deltas),
            reason=reason,
        ).as_dict()
        self._events.append(event)
        return {
            **event,
            "band_scores": list(event["band_scores"]),
            "band_weights": list(event["band_weights"]),
            "selected_layers": list(event["selected_layers"]),
            "violation_deltas": dict(event["violation_deltas"]),
        }
