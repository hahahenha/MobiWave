from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import copy
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import torch


@dataclass(frozen=True)
class DGLSSchedule:
    """Drift-conditioned slow-memory schedule used by one adaptation attempt."""

    slow_interval: int
    slow_weight: float
    shock: float
    persistence: float


class DGLSFastSlowOptimizer:
    """Layer-selective fast--slow optimizer from the DGLS update equations.

    Parameters are registered by their stable ``model.named_parameters()``
    names. ``layer_registry`` maps every layer identifier used by the selector
    to the parameter names owned by that layer. A parameter must belong to
    exactly one layer, which makes layer selection and optimizer checkpoints
    deterministic across cloned candidate models.

    The optimizer intentionally does not inherit :class:`torch.optim.Optimizer`.
    Its state is name-keyed rather than object-id-keyed, so an accepted
    optimizer checkpoint can be loaded atomically into an isolated model clone.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        layer_registry: Mapping[str, Iterable[str]] | None = None,
        lr: float = 3e-4,
        betas: tuple[float, float, float] = (0.9, 0.99, 0.999),
        base_slow_interval: int = 8,
        shock_scale: float = 1.0,
        persistence_scale: float = 1.0,
        interval_min: int = 1,
        interval_max: int = 64,
        omega_max: float = 0.35,
        newton_schulz_steps: int = 4,
        eps: float = 1e-8,
    ) -> None:
        parameters: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
        parameter_ids: set[int] = set()
        for name, parameter in named_parameters:
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be non-empty strings")
            if name in parameters:
                raise ValueError(f"Duplicate parameter name: {name}")
            if id(parameter) in parameter_ids:
                raise ValueError(f"Parameter registered more than once: {name}")
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(f"{name} is not a torch.nn.Parameter")
            parameters[name] = parameter
            parameter_ids.add(id(parameter))
        if not parameters:
            raise ValueError("named_parameters must contain at least one parameter")

        beta_fast, beta_slow, beta_second = betas
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must be finite numbers in [0, 1)")
        if (
            isinstance(base_slow_interval, bool)
            or not isinstance(base_slow_interval, Integral)
            or base_slow_interval < 1
        ):
            raise ValueError("base_slow_interval must be a positive integer")
        if (
            isinstance(interval_min, bool)
            or not isinstance(interval_min, Integral)
            or interval_min < 1
        ):
            raise ValueError("interval_min must be a positive integer")
        if (
            isinstance(interval_max, bool)
            or not isinstance(interval_max, Integral)
            or interval_max < interval_min
        ):
            raise ValueError("interval_max must be an integer no smaller than interval_min")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (shock_scale, persistence_scale)
        ):
            raise ValueError("Drift schedule scales must be finite and nonnegative")
        if not math.isfinite(omega_max) or not 0.0 <= omega_max <= 1.0:
            raise ValueError("omega_max must be a finite number in [0, 1]")
        if (
            isinstance(newton_schulz_steps, bool)
            or not isinstance(newton_schulz_steps, Integral)
            or newton_schulz_steps < 0
        ):
            raise ValueError("newton_schulz_steps must be a nonnegative integer")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self._parameters = parameters
        self._layers = self._normalize_registry(parameters, layer_registry)
        self._parameter_to_layer = {
            parameter_name: layer_name
            for layer_name, parameter_names in self._layers.items()
            for parameter_name in parameter_names
        }
        self.lr = float(lr)
        self.betas = (float(beta_fast), float(beta_slow), float(beta_second))
        self.base_slow_interval = int(base_slow_interval)
        self.shock_scale = float(shock_scale)
        self.persistence_scale = float(persistence_scale)
        self.interval_min = int(interval_min)
        self.interval_max = int(interval_max)
        self.omega_max = float(omega_max)
        self.newton_schulz_steps = int(newton_schulz_steps)
        self.eps = float(eps)

        self._selected_layers: tuple[str, ...] = ()
        self._layer_eta: dict[str, float] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self.global_inner_step = 0
        self.schedule = DGLSSchedule(
            slow_interval=self.base_slow_interval,
            slow_weight=0.0,
            shock=0.0,
            persistence=0.0,
        )

    @staticmethod
    def _normalize_registry(
        parameters: Mapping[str, torch.nn.Parameter],
        layer_registry: Mapping[str, Iterable[str]] | None,
    ) -> OrderedDict[str, tuple[str, ...]]:
        if layer_registry is None:
            generated: OrderedDict[str, list[str]] = OrderedDict()
            for parameter_name in parameters:
                layer_name = (
                    parameter_name.rsplit(".", 1)[0]
                    if "." in parameter_name
                    else parameter_name
                )
                generated.setdefault(layer_name, []).append(parameter_name)
            return OrderedDict((name, tuple(values)) for name, values in generated.items())

        normalized: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        seen: set[str] = set()
        for layer_name, layer_parameters in layer_registry.items():
            if not isinstance(layer_name, str) or not layer_name:
                raise ValueError("Layer names must be non-empty strings")
            if layer_name in normalized:
                raise ValueError(f"Duplicate layer name: {layer_name}")
            names = tuple(layer_parameters)
            if not names:
                raise ValueError(f"Layer {layer_name!r} contains no parameters")
            for parameter_name in names:
                if parameter_name not in parameters:
                    raise ValueError(
                        f"Layer {layer_name!r} references unknown parameter "
                        f"{parameter_name!r}"
                    )
                if parameter_name in seen:
                    raise ValueError(
                        f"Parameter {parameter_name!r} belongs to multiple layers"
                    )
                seen.add(parameter_name)
            normalized[layer_name] = names

        missing = tuple(name for name in parameters if name not in seen)
        if missing:
            raise ValueError(f"Layer registry does not cover parameters: {missing}")
        return normalized

    @property
    def layer_registry(self) -> OrderedDict[str, tuple[str, ...]]:
        return copy.deepcopy(self._layers)

    @property
    def selected_layers(self) -> tuple[str, ...]:
        return self._selected_layers

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self._parameters.values():
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def deactivate(self) -> None:
        """Mark every layer inactive while retaining accepted slow memories."""

        self._selected_layers = ()
        self._layer_eta = {}

    def configure(
        self,
        selected_layers: Iterable[str],
        *,
        eta_by_layer: Mapping[str, float] | None = None,
        shock: float,
        persistence: float,
        fixed_schedule: bool = False,
    ) -> DGLSSchedule:
        """Select layers, reactivate newly selected state, and set the schedule."""

        shock = self._nonnegative_finite("shock", shock)
        persistence = self._nonnegative_finite("persistence", persistence)
        if not isinstance(fixed_schedule, bool):
            raise TypeError("fixed_schedule must be boolean")
        requested = set(selected_layers)
        unknown = requested.difference(self._layers)
        if unknown:
            raise ValueError(f"Unknown selected layers: {tuple(sorted(unknown))}")
        selected = tuple(name for name in self._layers if name in requested)

        eta_source = {} if eta_by_layer is None else dict(eta_by_layer)
        extra_eta = set(eta_source).difference(selected)
        if extra_eta:
            raise ValueError(
                f"Learning rates were provided for unselected layers: "
                f"{tuple(sorted(extra_eta))}"
            )
        layer_eta: dict[str, float] = {}
        for layer_name in selected:
            eta = float(eta_source.get(layer_name, self.lr))
            if not math.isfinite(eta) or eta <= 0.0:
                raise ValueError(f"Learning rate for {layer_name!r} must be positive")
            layer_eta[layer_name] = eta

        newly_active = set(selected).difference(self._selected_layers)
        for layer_name in newly_active:
            for parameter_name in self._layers[layer_name]:
                state = self._state.get(parameter_name)
                if state is not None:
                    state["fast_momentum"].zero_()
                    state["second_moment"].zero_()
                    state["slow_accumulator"].zero_()
                    state["fast_step"] = 0
                    state["slow_counter"] = 0

        if fixed_schedule:
            interval = min(
                self.interval_max,
                max(self.interval_min, self.base_slow_interval),
            )
            slow_weight = self.omega_max
            schedule_shock = 0.0
            schedule_persistence = 0.0
        else:
            raw_interval = self.base_slow_interval * (
                1.0 + self.shock_scale * shock
            ) / (1.0 + self.persistence_scale * persistence)
            # All values are nonnegative. floor(x + 0.5) implements the
            # nearest-integer operator without Python's tie-to-even behaviour.
            interval = math.floor(raw_interval + 0.5)
            interval = min(self.interval_max, max(self.interval_min, interval))
            slow_weight = self.omega_max * persistence / (
                persistence + shock + self.eps
            )
            schedule_shock = shock
            schedule_persistence = persistence

        self._selected_layers = selected
        self._layer_eta = layer_eta
        self.schedule = DGLSSchedule(
            slow_interval=interval,
            slow_weight=slow_weight,
            shock=schedule_shock,
            persistence=schedule_persistence,
        )
        return self.schedule

    configure_drift = configure

    @staticmethod
    def _nonnegative_finite(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return result

    def _initialize_state(
        self, parameter_name: str, parameter: torch.nn.Parameter
    ) -> dict[str, Any]:
        state = {
            "fast_step": 0,
            "slow_write_count": 0,
            "slow_counter": 0,
            "fast_momentum": torch.zeros_like(parameter),
            "slow_momentum": torch.zeros_like(parameter),
            "second_moment": torch.zeros_like(parameter),
            "slow_accumulator": torch.zeros_like(parameter),
        }
        self._state[parameter_name] = state
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_inner_step += 1
        beta_fast, beta_slow, beta_second = self.betas
        selected = set(self._selected_layers)

        for parameter_name, parameter in self._parameters.items():
            layer_name = self._parameter_to_layer[parameter_name]
            if layer_name not in selected or parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("DGLSFastSlowOptimizer does not support sparse gradients")
            if gradient.shape != parameter.shape:
                raise RuntimeError(f"Gradient shape mismatch for {parameter_name}")
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient for {parameter_name}")

            state = self._state.get(parameter_name)
            if state is None:
                state = self._initialize_state(parameter_name, parameter)

            state["fast_step"] += 1
            fast_step = int(state["fast_step"])
            state["fast_momentum"].mul_(beta_fast).add_(
                gradient, alpha=1.0 - beta_fast
            )
            state["second_moment"].mul_(beta_second).addcmul_(
                gradient, gradient, value=1.0 - beta_second
            )
            state["slow_accumulator"].add_(gradient)
            state["slow_counter"] += 1

            if state["slow_counter"] >= self.schedule.slow_interval:
                mean_gradient = (
                    state["slow_accumulator"] / float(state["slow_counter"])
                )
                state["slow_momentum"].mul_(beta_slow).add_(
                    mean_gradient, alpha=1.0 - beta_slow
                )
                state["slow_accumulator"].zero_()
                state["slow_counter"] = 0
                state["slow_write_count"] += 1

            fast_hat = state["fast_momentum"] / (1.0 - beta_fast**fast_step)
            second_hat = state["second_moment"] / (
                1.0 - beta_second**fast_step
            )
            slow_writes = int(state["slow_write_count"])
            if slow_writes:
                slow_hat = state["slow_momentum"] / (
                    1.0 - beta_slow**slow_writes
                )
            else:
                slow_hat = torch.zeros_like(parameter)
            direction = (
                fast_hat + self.schedule.slow_weight * slow_hat
            ) / (second_hat.sqrt() + self.eps)
            if direction.ndim >= 2 and self.newton_schulz_steps:
                direction = _newton_schulz_direction(
                    direction, self.newton_schulz_steps, self.eps
                )
            parameter.add_(direction, alpha=-self._layer_eta[layer_name])

        return loss

    def parameter_state(
        self, parameter_name: str, *, clone: bool = True
    ) -> dict[str, Any] | None:
        if parameter_name not in self._parameters:
            raise KeyError(parameter_name)
        state = self._state.get(parameter_name)
        if state is None:
            return None
        return copy.deepcopy(state) if clone else state

    def state_dict(self) -> dict[str, Any]:
        """Return a detached deep snapshot suitable for atomic candidate state."""

        payload = {
            "version": self._STATE_VERSION,
            "parameter_names": tuple(self._parameters),
            "layer_registry": tuple(
                (name, tuple(parameters)) for name, parameters in self._layers.items()
            ),
            "hyperparameters": {
                "lr": self.lr,
                "betas": self.betas,
                "base_slow_interval": self.base_slow_interval,
                "shock_scale": self.shock_scale,
                "persistence_scale": self.persistence_scale,
                "interval_min": self.interval_min,
                "interval_max": self.interval_max,
                "omega_max": self.omega_max,
                "newton_schulz_steps": self.newton_schulz_steps,
                "eps": self.eps,
            },
            "selected_layers": self._selected_layers,
            "layer_eta": dict(self._layer_eta),
            "global_inner_step": self.global_inner_step,
            "schedule": {
                "slow_interval": self.schedule.slow_interval,
                "slow_weight": self.schedule.slow_weight,
                "shock": self.schedule.shock,
                "persistence": self.schedule.persistence,
            },
            "state": self._state,
        }
        return copy.deepcopy(payload)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Validate a full checkpoint before replacing any optimizer state."""

        payload = copy.deepcopy(dict(state_dict))
        if payload.get("version") != self._STATE_VERSION:
            raise ValueError("Unsupported DGLS optimizer state version")
        if tuple(payload.get("parameter_names", ())) != tuple(self._parameters):
            raise ValueError("Optimizer parameter registry does not match")
        expected_registry = tuple(
            (name, tuple(parameters)) for name, parameters in self._layers.items()
        )
        loaded_registry = tuple(
            (name, tuple(parameters))
            for name, parameters in payload.get("layer_registry", ())
        )
        if loaded_registry != expected_registry:
            raise ValueError("Optimizer layer registry does not match")
        if payload.get("hyperparameters") != self.state_dict()["hyperparameters"]:
            raise ValueError("Optimizer hyperparameters do not match")

        selected = tuple(payload.get("selected_layers", ()))
        if len(selected) != len(set(selected)) or any(
            layer not in self._layers for layer in selected
        ):
            raise ValueError("Invalid selected layer state")
        if selected != tuple(name for name in self._layers if name in selected):
            raise ValueError("Selected layers are not in stable registry order")
        layer_eta = dict(payload.get("layer_eta", {}))
        if set(layer_eta) != set(selected):
            raise ValueError("Layer learning-rate state does not match selection")
        for eta in layer_eta.values():
            if not math.isfinite(float(eta)) or float(eta) <= 0.0:
                raise ValueError("Invalid layer learning rate in state")

        global_inner_step = payload.get("global_inner_step")
        if (
            isinstance(global_inner_step, bool)
            or not isinstance(global_inner_step, int)
            or global_inner_step < 0
        ):
            raise ValueError("Invalid global inner-step counter")
        schedule_payload = dict(payload.get("schedule", {}))
        schedule = DGLSSchedule(
            slow_interval=int(schedule_payload["slow_interval"]),
            slow_weight=float(schedule_payload["slow_weight"]),
            shock=float(schedule_payload["shock"]),
            persistence=float(schedule_payload["persistence"]),
        )
        if not self.interval_min <= schedule.slow_interval <= self.interval_max:
            raise ValueError("Invalid slow interval in state")
        if (
            not math.isfinite(schedule.slow_weight)
            or not 0.0 <= schedule.slow_weight <= self.omega_max
        ):
            raise ValueError("Invalid slow weight in state")
        self._nonnegative_finite("shock", schedule.shock)
        self._nonnegative_finite("persistence", schedule.persistence)

        loaded_state = dict(payload.get("state", {}))
        if not set(loaded_state).issubset(self._parameters):
            raise ValueError("Optimizer state contains an unknown parameter")
        materialized: dict[str, dict[str, Any]] = {}
        tensor_keys = (
            "fast_momentum",
            "slow_momentum",
            "second_moment",
            "slow_accumulator",
        )
        counter_keys = ("fast_step", "slow_write_count", "slow_counter")
        for parameter_name, raw_state in loaded_state.items():
            parameter = self._parameters[parameter_name]
            current = dict(raw_state)
            if set(current) != set(tensor_keys).union(counter_keys):
                raise ValueError(f"Malformed optimizer state for {parameter_name}")
            normalized: dict[str, Any] = {}
            for key in tensor_keys:
                tensor = current[key]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape:
                    raise ValueError(
                        f"Invalid {key} state shape for {parameter_name}"
                    )
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite {key} state for {parameter_name}")
                normalized[key] = tensor.detach().to(
                    device=parameter.device, dtype=parameter.dtype
                ).clone()
            for key in counter_keys:
                value = current[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"Invalid {key} for {parameter_name}")
                normalized[key] = value
            materialized[parameter_name] = normalized

        # Commit only after every field and tensor has been validated.
        self._selected_layers = selected
        self._layer_eta = {name: float(value) for name, value in layer_eta.items()}
        self.global_inner_step = global_inner_step
        self.schedule = schedule
        self._state = materialized


def _newton_schulz_direction(
    direction: torch.Tensor, steps: int, eps: float
) -> torch.Tensor:
    """Apply the paper's Newton--Schulz map to a matrix-shaped direction."""

    original_shape = direction.shape
    matrix = direction.reshape(direction.shape[0], -1)
    if matrix.numel() == 0:
        return direction
    work = matrix.float()
    norm = work.norm()
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        return direction
    work = work / (norm + eps)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.t()
    for _ in range(steps):
        work = 1.5 * work - 0.5 * (work @ work.t()) @ work
    if transposed:
        work = work.t()
    if not torch.isfinite(work).all():
        return direction
    return work.to(dtype=direction.dtype).reshape(original_shape)
