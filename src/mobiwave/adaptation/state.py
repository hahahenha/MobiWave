from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import copy
import math
import pickle

import numpy as np
import torch


class _FrozenPayload:
    """Pickle-backed immutable payload that is cheap and safe to share."""

    __slots__ = ("_data",)

    def __init__(self, value: Any) -> None:
        object.__setattr__(
            self,
            "_data",
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("_FrozenPayload is immutable")

    def thaw(self) -> Any:
        return pickle.loads(self._data)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenPayload":
        memo[id(self)] = self
        return self


def deep_clone(
    value: Any,
    _memo: dict[int, Any] | None = None,
) -> Any:
    """Clone transaction state without sharing mutable arrays or tensors."""

    memo = {} if _memo is None else _memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if torch.is_tensor(value):
        output = value.detach().clone()
        memo[identity] = output
        return output
    if isinstance(value, np.ndarray):
        output = value.copy()
        memo[identity] = output
        return output
    if isinstance(value, Mapping):
        output: dict[Any, Any] = {}
        memo[identity] = output
        output.update(
            {
                deep_clone(key, memo): deep_clone(item, memo)
                for key, item in value.items()
            }
        )
        return output
    if isinstance(value, tuple):
        output = tuple(deep_clone(item, memo) for item in value)
        memo[identity] = output
        return output
    if isinstance(value, list):
        output_list: list[Any] = []
        memo[identity] = output_list
        output_list.extend(deep_clone(item, memo) for item in value)
        return output_list
    if isinstance(value, set):
        output_set: set[Any] = set()
        memo[identity] = output_set
        output_set.update(deep_clone(item, memo) for item in value)
        return output_set
    return copy.deepcopy(value, memo)


@dataclass
class MutableState:
    """Materialized state passed to an isolated candidate callback."""

    model_state: Any
    optimizer_state: Any
    importance_state: Any
    reference_state: Any
    version: int

    def clone(self) -> "MutableState":
        return MutableState(
            model_state=deep_clone(self.model_state),
            optimizer_state=deep_clone(self.optimizer_state),
            importance_state=deep_clone(self.importance_state),
            reference_state=deep_clone(self.reference_state),
            version=int(self.version),
        )


class AcceptedState:
    """Immutable-by-interface snapshot of the last accepted model transaction.

    Every value is cloned on ingress and egress.  A callback can therefore
    mutate a materialized candidate without changing the accepted snapshot.
    """

    __slots__ = (
        "_model_state",
        "_optimizer_state",
        "_importance_state",
        "_reference_state",
        "_version",
    )

    def __init__(
        self,
        *,
        model_state: Any,
        optimizer_state: Any,
        importance_state: Any,
        reference_state: Any,
        version: int = 0,
    ) -> None:
        if isinstance(version, bool) or not isinstance(version, Integral) or int(version) < 0:
            raise ValueError("AcceptedState version must be a non-negative integer")
        object.__setattr__(self, "_model_state", deep_clone(model_state))
        object.__setattr__(self, "_optimizer_state", deep_clone(optimizer_state))
        object.__setattr__(self, "_importance_state", deep_clone(importance_state))
        frozen_reference = (
            reference_state
            if isinstance(reference_state, _FrozenPayload)
            else _FrozenPayload(reference_state)
        )
        object.__setattr__(self, "_reference_state", frozen_reference)
        object.__setattr__(self, "_version", int(version))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("AcceptedState is immutable")

    @property
    def model_state(self) -> Any:
        return deep_clone(self._model_state)

    @property
    def optimizer_state(self) -> Any:
        return deep_clone(self._optimizer_state)

    @property
    def importance_state(self) -> Any:
        return deep_clone(self._importance_state)

    @property
    def reference_state(self) -> Any:
        return self._reference_state.thaw()

    @property
    def version(self) -> int:
        return self._version

    def materialize(self) -> MutableState:
        return MutableState(
            model_state=self.model_state,
            optimizer_state=self.optimizer_state,
            importance_state=self.importance_state,
            reference_state=self.reference_state,
            version=self.version,
        )

    def clone(self) -> "AcceptedState":
        return AcceptedState(
            model_state=self._model_state,
            optimizer_state=self._optimizer_state,
            importance_state=self._importance_state,
            reference_state=self._reference_state,
            version=self._version,
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: "CandidateState",
        *,
        version: int,
    ) -> "AcceptedState":
        return cls(
            model_state=candidate._model_state,
            optimizer_state=candidate._optimizer_state,
            importance_state=candidate._importance_state,
            reference_state=candidate._reference_state,
            version=version,
        )

    def with_version(self, version: int) -> "AcceptedState":
        return AcceptedState(
            model_state=self._model_state,
            optimizer_state=self._optimizer_state,
            importance_state=self._importance_state,
            reference_state=self._reference_state,
            version=version,
        )


class CandidateState:
    """Frozen candidate payload produced from a materialized accepted state."""

    __slots__ = (
        "_model_state",
        "_optimizer_state",
        "_importance_state",
        "_reference_state",
    )

    def __init__(
        self,
        *,
        model_state: Any,
        optimizer_state: Any,
        importance_state: Any,
        reference_state: Any,
    ) -> None:
        object.__setattr__(self, "_model_state", deep_clone(model_state))
        object.__setattr__(self, "_optimizer_state", deep_clone(optimizer_state))
        object.__setattr__(self, "_importance_state", deep_clone(importance_state))
        frozen_reference = (
            reference_state
            if isinstance(reference_state, _FrozenPayload)
            else _FrozenPayload(reference_state)
        )
        object.__setattr__(self, "_reference_state", frozen_reference)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("CandidateState is immutable")

    @property
    def model_state(self) -> Any:
        return deep_clone(self._model_state)

    @property
    def optimizer_state(self) -> Any:
        return deep_clone(self._optimizer_state)

    @property
    def importance_state(self) -> Any:
        return deep_clone(self._importance_state)

    @property
    def reference_state(self) -> Any:
        return self._reference_state.thaw()

    @classmethod
    def from_mutable(cls, state: MutableState) -> "CandidateState":
        return cls(
            model_state=state.model_state,
            optimizer_state=state.optimizer_state,
            importance_state=state.importance_state,
            reference_state=state.reference_state,
        )

    def materialize(self, *, version: int) -> MutableState:
        return MutableState(
            model_state=self.model_state,
            optimizer_state=self.optimizer_state,
            importance_state=self.importance_state,
            reference_state=self.reference_state,
            version=version,
        )

    def clone(self) -> "CandidateState":
        return CandidateState(
            model_state=self._model_state,
            optimizer_state=self._optimizer_state,
            importance_state=self._importance_state,
            reference_state=self._reference_state,
        )


def _default_time_getter(record: Any) -> int:
    if isinstance(record, Mapping) and "time" in record:
        value = record["time"]
    elif hasattr(record, "time"):
        value = getattr(record, "time")
    else:
        raise TypeError("Each causal record must expose a 'time' field or attribute")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Causal record times must be integers")
    return int(value)


class CausalSplit:
    """Chronological adaptation/validation split whose records are strictly < t."""

    __slots__ = (
        "_adaptation",
        "_validation",
        "_adaptation_times",
        "_validation_times",
        "_decision_time",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("CausalSplit is immutable")

    def __init__(
        self,
        *,
        adaptation: Sequence[Any],
        validation: Sequence[Any],
        decision_time: int,
        time_getter: Callable[[Any], int] | None = None,
    ) -> None:
        if (
            isinstance(decision_time, bool)
            or not isinstance(decision_time, Integral)
            or int(decision_time) < 0
        ):
            raise ValueError("decision_time must be a non-negative integer")
        getter = time_getter or _default_time_getter
        adaptation_values = tuple(adaptation)
        validation_values = tuple(validation)
        if not adaptation_values:
            raise ValueError("CausalSplit requires at least one adaptation record")
        if not validation_values:
            raise ValueError("CausalSplit requires at least one validation record")
        adaptation_times = tuple(getter(record) for record in adaptation_values)
        validation_times = tuple(getter(record) for record in validation_values)
        all_times = adaptation_times + validation_times
        if any(time >= int(decision_time) for time in all_times):
            raise ValueError("Every CausalSplit record must satisfy record.time < decision_time")
        if tuple(sorted(adaptation_times)) != adaptation_times:
            raise ValueError("Adaptation records must be chronological")
        if tuple(sorted(validation_times)) != validation_times:
            raise ValueError("Validation records must be chronological")
        if max(adaptation_times) >= min(validation_times):
            raise ValueError("Validation records must be strictly later than adaptation records")
        object.__setattr__(
            self,
            "_adaptation",
            _FrozenPayload(adaptation_values),
        )
        object.__setattr__(
            self,
            "_validation",
            _FrozenPayload(validation_values),
        )
        object.__setattr__(self, "_adaptation_times", adaptation_times)
        object.__setattr__(self, "_validation_times", validation_times)
        object.__setattr__(self, "_decision_time", int(decision_time))

    @classmethod
    def from_records(
        cls,
        records: Sequence[Any],
        *,
        decision_time: int,
        validation_size: int,
        time_getter: Callable[[Any], int] | None = None,
    ) -> "CausalSplit":
        if (
            isinstance(validation_size, bool)
            or not isinstance(validation_size, Integral)
            or int(validation_size) < 1
        ):
            raise ValueError("validation_size must be a positive integer")
        getter = time_getter or _default_time_getter
        ordered = sorted(records, key=getter)
        if len(ordered) <= int(validation_size):
            raise ValueError("CausalSplit needs adaptation records before the validation suffix")
        cut = len(ordered) - int(validation_size)
        return cls(
            adaptation=ordered[:cut],
            validation=ordered[cut:],
            decision_time=decision_time,
            time_getter=getter,
        )

    @property
    def adaptation(self) -> tuple[Any, ...]:
        return tuple(self._adaptation.thaw())

    @property
    def validation(self) -> tuple[Any, ...]:
        return tuple(self._validation.thaw())

    @property
    def adaptation_times(self) -> tuple[int, ...]:
        return self._adaptation_times

    @property
    def validation_times(self) -> tuple[int, ...]:
        return self._validation_times

    @property
    def decision_time(self) -> int:
        return self._decision_time

    def clone(self) -> "CausalSplit":
        output = object.__new__(CausalSplit)
        object.__setattr__(output, "_adaptation", self._adaptation)
        object.__setattr__(output, "_validation", self._validation)
        object.__setattr__(
            output,
            "_adaptation_times",
            self._adaptation_times,
        )
        object.__setattr__(
            output,
            "_validation_times",
            self._validation_times,
        )
        object.__setattr__(output, "_decision_time", self._decision_time)
        return output


@dataclass(frozen=True)
class LayerSignals:
    activation: Mapping[str, float]
    gradient: Mapping[str, float]
    importance: Mapping[str, float]
    costs: Mapping[str, float]

    def __post_init__(self) -> None:
        names = tuple(sorted(self.costs))
        if not names:
            raise ValueError("LayerSignals requires at least one layer cost")
        checked_costs = {name: float(self.costs[name]) for name in names}
        if any(not math.isfinite(value) or value <= 0.0 for value in checked_costs.values()):
            raise ValueError("Layer costs must be finite and positive")
        for mapping in (self.activation, self.gradient, self.importance):
            unknown = set(mapping) - set(names)
            if unknown:
                raise ValueError(f"Layer signal contains unknown layers: {sorted(unknown)}")
            if any(not math.isfinite(float(value)) for value in mapping.values()):
                raise ValueError("Layer signals must be finite")
        object.__setattr__(
            self,
            "activation",
            MappingProxyType(
                {name: float(self.activation.get(name, 0.0)) for name in names}
            ),
        )
        object.__setattr__(
            self,
            "gradient",
            MappingProxyType(
                {name: float(self.gradient.get(name, 0.0)) for name in names}
            ),
        )
        object.__setattr__(
            self,
            "importance",
            MappingProxyType(
                {name: float(self.importance.get(name, 0.0)) for name in names}
            ),
        )
        object.__setattr__(self, "costs", MappingProxyType(checked_costs))
