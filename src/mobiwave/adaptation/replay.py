from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import copy
import hashlib
import math
import pickle

from .dgls import (
    CandidateMetrics,
    CandidateValidationResult,
    validate_candidate,
)


def _serialize(value: Any) -> bytes:
    """Serialize one trusted, local replay value into an immutable blob."""

    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _combined_digest(*values: str) -> str:
    return _digest("\0".join(values).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class EnvCheckpoint:
    """Immutable snapshot from which both suffix replays must start.

    ``state_blob`` deliberately contains no reference to the live environment.
    Every call to :meth:`restore_state` returns a fresh object.
    """

    time: int
    state_blob: bytes = field(repr=False)
    initial_state_hash: str
    metadata_hash: str

    @classmethod
    def capture(
        cls,
        state: Any,
        *,
        time: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EnvCheckpoint":
        if isinstance(time, bool) or int(time) < 0:
            raise ValueError("checkpoint time must be a nonnegative integer")
        time = int(time)
        state_blob = _serialize(state)
        initial_state_hash = _digest(state_blob)
        metadata_blob = _serialize(
            {
                "time": time,
                "initial_state_hash": initial_state_hash,
                "metadata": dict(metadata or {}),
            }
        )
        return cls(
            time=time,
            state_blob=state_blob,
            initial_state_hash=initial_state_hash,
            metadata_hash=_digest(metadata_blob),
        )

    def restore_state(self) -> Any:
        return pickle.loads(self.state_blob)


@dataclass(frozen=True, slots=True)
class ExogenousTape:
    """Immutable requests, travel times, and topology for a held-out suffix."""

    start_time: int
    end_time: int
    request_blob: bytes = field(repr=False)
    travel_blob: bytes = field(repr=False)
    topology_blob: bytes = field(repr=False)
    request_hash: str
    travel_hash: str
    topology_hash: str
    metadata_hash: str

    @classmethod
    def capture(
        cls,
        *,
        start_time: int,
        end_time: int,
        requests: Any,
        travel_times: Any,
        topology: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExogenousTape":
        if (
            isinstance(start_time, bool)
            or isinstance(end_time, bool)
            or int(start_time) < 0
            or int(end_time) <= int(start_time)
        ):
            raise ValueError("suffix requires integer times with 0 <= start_time < end_time")
        start_time = int(start_time)
        end_time = int(end_time)
        request_blob = _serialize(requests)
        travel_blob = _serialize(travel_times)
        topology_blob = _serialize(topology)
        request_hash = _digest(request_blob)
        travel_hash = _digest(travel_blob)
        topology_hash = _digest(topology_blob)
        metadata_blob = _serialize(
            {
                "start_time": start_time,
                "end_time": end_time,
                "request_hash": request_hash,
                "travel_hash": travel_hash,
                "topology_hash": topology_hash,
                "metadata": dict(metadata or {}),
            }
        )
        return cls(
            start_time=start_time,
            end_time=end_time,
            request_blob=request_blob,
            travel_blob=travel_blob,
            topology_blob=topology_blob,
            request_hash=request_hash,
            travel_hash=travel_hash,
            topology_hash=topology_hash,
            metadata_hash=_digest(metadata_blob),
        )

    @property
    def exogenous_hash(self) -> str:
        return _combined_digest(
            self.request_hash,
            self.travel_hash,
            self.topology_hash,
        )

    def restore_requests(self) -> Any:
        return pickle.loads(self.request_blob)

    def restore_travel_times(self) -> Any:
        return pickle.loads(self.travel_blob)

    def restore_topology(self) -> Any:
        return pickle.loads(self.topology_blob)


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """Metrics and replay provenance returned by one isolated suffix replay."""

    reward: float
    violations: Mapping[str, float]
    initial_state_hash: str
    request_hash: str
    travel_hash: str
    topology_hash: str

    def __post_init__(self) -> None:
        immutable = MappingProxyType(
            {str(name): float(value) for name, value in self.violations.items()}
        )
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "violations", immutable)
        for field_name in (
            "initial_state_hash",
            "request_hash",
            "travel_hash",
            "topology_hash",
        ):
            object.__setattr__(self, field_name, str(getattr(self, field_name)))

    @property
    def exogenous_hash(self) -> str:
        return _combined_digest(
            self.request_hash,
            self.travel_hash,
            self.topology_hash,
        )

    def candidate_metrics(self) -> CandidateMetrics:
        return CandidateMetrics(
            reward=self.reward,
            violations=dict(self.violations),
        )


@dataclass(frozen=True, slots=True)
class PairedReplayResult:
    """Outcome of validating one candidate on a matched held-out suffix."""

    available: bool
    accepted: bool
    reason: str
    stable: ReplayOutcome | None = None
    candidate: ReplayOutcome | None = None
    validation: CandidateValidationResult | None = None
    error: str | None = None

    @property
    def reward_gain(self) -> float | None:
        return None if self.validation is None else self.validation.reward_gain

    @property
    def violation_deltas(self) -> Mapping[str, float]:
        if self.validation is None:
            return MappingProxyType({})
        return MappingProxyType(dict(self.validation.violation_deltas))


ReplayValue = ReplayOutcome | Mapping[str, Any] | tuple[Mapping[str, Any], Mapping[str, Any]]
ReplayFn = Callable[[EnvCheckpoint, Any, ExogenousTape], ReplayValue]


class PairedSuffixValidator:
    """Validate stable and candidate states on independently restored replays."""

    def __init__(
        self,
        replay_fn: ReplayFn,
        *,
        required_violations: Iterable[str],
        reward_margin: float = 0.0,
        violation_tolerances: Mapping[str, float] | None = None,
    ) -> None:
        if not callable(replay_fn):
            raise TypeError("replay_fn must be callable")
        names = tuple(str(name).strip() for name in required_violations)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("required_violations must contain unique, nonempty names")
        if (
            not math.isfinite(float(reward_margin))
            or float(reward_margin) < 0.0
        ):
            raise ValueError("reward_margin must be finite and nonnegative")
        tolerances = {
            str(name): float(value)
            for name, value in dict(violation_tolerances or {}).items()
        }
        unknown = set(tolerances) - set(names)
        if unknown:
            raise ValueError(
                "violation tolerances contain unknown metrics: "
                + ",".join(sorted(unknown))
            )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in tolerances.values()
        ):
            raise ValueError("violation tolerances must be finite and nonnegative")

        self.replay_fn = replay_fn
        self.required_violations = names
        self.reward_margin = float(reward_margin)
        self.violation_tolerances = {
            name: tolerances.get(name, 0.0) for name in names
        }

    def validate(
        self,
        checkpoint: EnvCheckpoint,
        stable_model_state: Any,
        candidate_model_state: Any,
        suffix: ExogenousTape,
        *,
        evaluation_order: Sequence[str] = ("stable", "candidate"),
    ) -> PairedReplayResult:
        order = tuple(evaluation_order)
        if len(order) != 2 or set(order) != {"stable", "candidate"}:
            raise ValueError(
                "evaluation_order must contain stable and candidate exactly once"
            )
        if checkpoint.time != suffix.start_time:
            return PairedReplayResult(
                available=False,
                accepted=False,
                reason="checkpoint_suffix_mismatch",
                error=(
                    f"checkpoint time {checkpoint.time} does not match "
                    f"suffix start {suffix.start_time}"
                ),
            )

        model_states = {
            "stable": stable_model_state,
            "candidate": candidate_model_state,
        }
        outcomes: dict[str, ReplayOutcome] = {}
        try:
            for label in order:
                raw = self.replay_fn(
                    checkpoint,
                    copy.deepcopy(model_states[label]),
                    suffix,
                )
                outcomes[label] = self._coerce_outcome(raw)
        except Exception as exc:
            return PairedReplayResult(
                available=False,
                accepted=False,
                reason="validation_error",
                error=f"{type(exc).__name__}: {exc}",
            )

        stable = outcomes["stable"]
        candidate = outcomes["candidate"]
        provenance_error = self._provenance_error(
            checkpoint,
            suffix,
            stable,
            candidate,
        )
        if provenance_error is not None:
            return PairedReplayResult(
                available=False,
                accepted=False,
                reason="replay_mismatch",
                stable=stable,
                candidate=candidate,
                error=provenance_error,
            )

        schema_error = self._schema_error(stable, candidate)
        if schema_error is not None:
            return PairedReplayResult(
                available=False,
                accepted=False,
                reason="invalid_outcome",
                stable=stable,
                candidate=candidate,
                error=schema_error,
            )

        try:
            validation = validate_candidate(
                stable.candidate_metrics(),
                candidate.candidate_metrics(),
                reward_margin=self.reward_margin,
                violation_tolerances=self.violation_tolerances,
                required_violations=self.required_violations,
            )
        except Exception as exc:
            return PairedReplayResult(
                available=False,
                accepted=False,
                reason="validation_error",
                stable=stable,
                candidate=candidate,
                error=f"{type(exc).__name__}: {exc}",
            )
        return PairedReplayResult(
            available=True,
            accepted=validation.accepted,
            reason=validation.reason,
            stable=stable,
            candidate=candidate,
            validation=validation,
        )

    @staticmethod
    def _coerce_outcome(value: ReplayValue) -> ReplayOutcome:
        if isinstance(value, ReplayOutcome):
            return value
        if isinstance(value, tuple):
            if len(value) != 2:
                raise TypeError("replay tuple must contain metrics and hashes")
            metrics, hashes = value
            if not isinstance(metrics, Mapping) or not isinstance(hashes, Mapping):
                raise TypeError("replay tuple values must both be mappings")
            payload = {**dict(metrics), **dict(hashes)}
        elif isinstance(value, Mapping):
            payload = dict(value)
            metrics = payload.pop("metrics", None)
            hashes = payload.pop("hashes", None)
            if metrics is not None:
                if not isinstance(metrics, Mapping):
                    raise TypeError("replay metrics must be a mapping")
                payload.update(metrics)
            if hashes is not None:
                if not isinstance(hashes, Mapping):
                    raise TypeError("replay hashes must be a mapping")
                payload.update(hashes)
        else:
            raise TypeError("replay_fn must return ReplayOutcome or metric/hash mappings")
        return ReplayOutcome(
            reward=payload["reward"],
            violations=payload["violations"],
            initial_state_hash=payload["initial_state_hash"],
            request_hash=payload["request_hash"],
            travel_hash=payload["travel_hash"],
            topology_hash=payload["topology_hash"],
        )

    def _schema_error(
        self,
        stable: ReplayOutcome,
        candidate: ReplayOutcome,
    ) -> str | None:
        required = set(self.required_violations)
        for label, outcome in (("stable", stable), ("candidate", candidate)):
            actual = set(outcome.violations)
            if actual != required:
                missing = ",".join(sorted(required - actual)) or "-"
                extra = ",".join(sorted(actual - required)) or "-"
                return f"{label} violation schema mismatch; missing={missing}; extra={extra}"
            values = (outcome.reward, *outcome.violations.values())
            if any(not math.isfinite(float(value)) for value in values):
                return f"{label} replay metrics must all be finite"
        return None

    @staticmethod
    def _provenance_error(
        checkpoint: EnvCheckpoint,
        suffix: ExogenousTape,
        stable: ReplayOutcome,
        candidate: ReplayOutcome,
    ) -> str | None:
        expected = {
            "initial_state_hash": checkpoint.initial_state_hash,
            "request_hash": suffix.request_hash,
            "travel_hash": suffix.travel_hash,
            "topology_hash": suffix.topology_hash,
        }
        mismatches = []
        for field_name, expected_value in expected.items():
            stable_value = getattr(stable, field_name)
            candidate_value = getattr(candidate, field_name)
            if (
                stable_value != expected_value
                or candidate_value != expected_value
                or stable_value != candidate_value
            ):
                mismatches.append(field_name)
        if mismatches:
            return "paired replay hashes differ: " + ",".join(mismatches)
        return None


__all__ = [
    "EnvCheckpoint",
    "ExogenousTape",
    "PairedReplayResult",
    "PairedSuffixValidator",
    "ReplayFn",
    "ReplayOutcome",
]
