from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from types import MappingProxyType
from typing import Mapping
import math

import torch


def deterministic_row_sample(
    values: torch.Tensor,
    sample_cap: int | None,
) -> torch.Tensor:
    """Select bounded, evenly spaced rows without consuming random state."""

    if sample_cap is None or values.shape[0] <= sample_cap:
        return values
    if isinstance(sample_cap, bool) or int(sample_cap) < 2:
        raise ValueError("sample_cap must be at least two")
    count = int(values.shape[0])
    cap = int(sample_cap)
    # Pick each equal-width interval's midpoint.  The integer expression is
    # deterministic on every device and cannot create duplicate indices when
    # cap < count.
    positions = torch.arange(cap, device=values.device, dtype=torch.long)
    indices = ((2 * positions + 1) * count) // (2 * cap)
    return values.index_select(0, indices)


def _rbf_kernel_sum(
    left: torch.Tensor,
    right: torch.Tensor,
    sigma: torch.Tensor,
    *,
    block_size: int = 512,
) -> torch.Tensor:
    """Accumulate an exact RBF kernel sum with bounded working memory."""

    total = left.new_zeros(())
    denominator = 2.0 * sigma.square()
    for left_start in range(0, left.shape[0], block_size):
        left_block = left[left_start : left_start + block_size]
        for right_start in range(0, right.shape[0], block_size):
            right_block = right[right_start : right_start + block_size]
            squared_distance = torch.cdist(left_block, right_block).square()
            total = total + torch.exp(-squared_distance / denominator).sum()
    return total


def rbf_mmd2(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidth: float | None = None,
    *,
    sample_cap: int | None = None,
) -> torch.Tensor:
    """Biased squared MMD, evaluated in exact memory-bounded kernel blocks."""

    left = left.reshape(left.shape[0], -1).float()
    right = right.reshape(right.shape[0], -1).float()
    if left.shape[0] == 0 or right.shape[0] == 0:
        raise ValueError("MMD requires at least one sample in each set")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("MMD inputs must be finite")
    left = deterministic_row_sample(left, sample_cap)
    right = deterministic_row_sample(right, sample_cap)
    if bandwidth is None:
        combined = torch.cat([left, right], dim=0)
        distances = torch.pdist(combined)
        positive = distances[distances > 0]
        bandwidth_tensor = (
            positive.median()
            if positive.numel()
            else left.new_tensor(1.0)
        )
    else:
        if not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0.0:
            raise ValueError("RBF bandwidth must be finite and positive")
        bandwidth_tensor = torch.tensor(
            float(bandwidth),
            dtype=left.dtype,
            device=left.device,
        )
    sigma = bandwidth_tensor.clamp_min(torch.finfo(left.dtype).eps)
    k_xx = _rbf_kernel_sum(left, left, sigma) / (left.shape[0] ** 2)
    k_yy = _rbf_kernel_sum(right, right, sigma) / (right.shape[0] ** 2)
    k_xy = _rbf_kernel_sum(left, right, sigma) / (
        left.shape[0] * right.shape[0]
    )
    return (k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0)


def dispatch_weighted_spectral_drift(
    recent_features: torch.Tensor,
    reference_features: torch.Tensor,
    recent_gate_weights: torch.Tensor,
    bandwidth: float | Sequence[float] | None = None,
    sample_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gate-weighted graph-frequency drift.

    Feature tensors have shape ``[..., K, F]`` and gate weights have shape
    ``[..., K]``.  All leading sample and node dimensions are flattened per
    band, matching the empirical MMD in the paper.
    """
    if recent_features.ndim < 3 or reference_features.ndim < 3:
        raise ValueError("Spectral features must include sample, band, and feature dimensions")
    if recent_features.shape[-2:] != reference_features.shape[-2:]:
        raise ValueError("Recent and reference spectral band dimensions must match")
    band_count = recent_features.shape[-2]
    if tuple(recent_gate_weights.shape) != tuple(recent_features.shape[:-1]):
        raise ValueError("Gate weights must match every recent sample/node and frequency band")
    if (
        not torch.isfinite(recent_gate_weights).all()
        or torch.any(recent_gate_weights < 0)
        or float(recent_gate_weights.sum().detach().cpu().item()) <= 0.0
    ):
        raise ValueError("Gate weights must be finite, nonnegative, and have positive mass")
    if isinstance(bandwidth, Sequence) and not isinstance(bandwidth, (str, bytes)):
        band_bandwidths = tuple(float(value) for value in bandwidth)
        if len(band_bandwidths) != band_count:
            raise ValueError("One fixed RBF bandwidth is required per spectral band")
    else:
        band_bandwidths = (bandwidth,) * band_count
    feature_width = recent_features.shape[-1]
    recent_flat = recent_features.reshape(-1, band_count, feature_width)
    reference_flat = reference_features.reshape(-1, band_count, feature_width)
    gates = recent_gate_weights.reshape(-1, band_count)
    band_weights = gates.mean(dim=0)
    band_weights = band_weights / band_weights.sum().clamp_min(torch.finfo(gates.dtype).eps)
    band_drifts = torch.stack(
        [
            rbf_mmd2(
                recent_flat[:, band, :],
                reference_flat[:, band, :],
                band_bandwidths[band],
                sample_cap=sample_cap,
            )
            for band in range(band_count)
        ]
    )
    score = torch.sum(band_weights * band_drifts)
    return score, band_drifts, band_weights


def hysteresis_state(
    score: float,
    previous: bool,
    threshold_on: float,
    threshold_off: float,
) -> bool:
    if not (
        math.isfinite(score)
        and math.isfinite(threshold_on)
        and math.isfinite(threshold_off)
        and threshold_on > threshold_off >= 0.0
    ):
        raise ValueError("Hysteresis requires finite score and threshold_on > threshold_off >= 0")
    if score >= threshold_on:
        return True
    if score <= threshold_off:
        return False
    return bool(previous)


@dataclass(frozen=True)
class LayerSelection:
    selected: tuple[str, ...]
    scores: Mapping[str, float]
    normalized_activation: Mapping[str, float]
    normalized_gradient: Mapping[str, float]
    normalized_importance: Mapping[str, float]
    selected_cost: float
    budget: float

    def __post_init__(self) -> None:
        for name in (
            "scores",
            "normalized_activation",
            "normalized_gradient",
            "normalized_importance",
        ):
            values = {
                str(key): float(value)
                for key, value in getattr(self, name).items()
            }
            object.__setattr__(self, name, MappingProxyType(values))


def _minmax(values: Mapping[str, float], names: tuple[str, ...]) -> dict[str, float]:
    checked = {name: float(values.get(name, 0.0)) for name in names}
    if any(not math.isfinite(value) for value in checked.values()):
        raise ValueError("Layer signals must be finite")
    low = min(checked.values(), default=0.0)
    high = max(checked.values(), default=0.0)
    if high <= low:
        return {name: 0.0 for name in names}
    epsilon = 1e-12
    return {
        name: (value - low) / (high - low + epsilon)
        for name, value in checked.items()
    }


def select_layers_under_budget(
    activation: Mapping[str, float],
    gradient: Mapping[str, float],
    importance: Mapping[str, float],
    costs: Mapping[str, float],
    budget_ratio: float,
    *,
    weight_activation: float = 0.45,
    weight_gradient: float = 0.45,
    weight_importance: float = 0.10,
) -> LayerSelection:
    names = tuple(sorted(costs))
    if not names:
        raise ValueError("Layer selection requires at least one layer")
    checked_costs = {name: float(costs[name]) for name in names}
    if any(not math.isfinite(cost) or cost <= 0.0 for cost in checked_costs.values()):
        raise ValueError("Every layer cost must be finite and positive")
    if not math.isfinite(float(budget_ratio)) or not 0.0 < float(budget_ratio) <= 1.0:
        raise ValueError("budget_ratio must be in (0, 1]")
    weights = (weight_activation, weight_gradient, weight_importance)
    if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in weights):
        raise ValueError("Layer-score weights must be finite and nonnegative")
    norm_activation = _minmax(activation, names)
    norm_gradient = _minmax(gradient, names)
    norm_importance = _minmax(importance, names)
    scores = {
        name: (
            weight_activation * norm_activation[name]
            + weight_gradient * norm_gradient[name]
            - weight_importance * norm_importance[name]
        )
        for name in names
    }
    budget = float(budget_ratio) * sum(checked_costs.values())
    ranked = sorted(
        (
            (scores[name] / checked_costs[name], scores[name], name)
            for name in names
            if scores[name] > 0.0
        ),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    selected = []
    selected_cost = 0.0
    for _, _, name in ranked:
        cost = checked_costs[name]
        if selected_cost + cost <= budget + 1e-12:
            selected.append(name)
            selected_cost += cost
    return LayerSelection(
        selected=tuple(selected),
        scores=scores,
        normalized_activation=norm_activation,
        normalized_gradient=norm_gradient,
        normalized_importance=norm_importance,
        selected_cost=selected_cost,
        budget=budget,
    )


@dataclass(frozen=True)
class CandidateMetrics:
    reward: float
    violations: Mapping[str, float]


@dataclass(frozen=True)
class CandidateValidationResult:
    accepted: bool
    reward_gain: float
    violation_deltas: Mapping[str, float]
    reason: str


def validate_candidate(
    stable: CandidateMetrics,
    candidate: CandidateMetrics,
    *,
    reward_margin: float = 0.0,
    violation_tolerances: Mapping[str, float] | None = None,
    required_violations: Sequence[str] | None = None,
) -> CandidateValidationResult:
    if not math.isfinite(float(reward_margin)) or float(reward_margin) < 0.0:
        raise ValueError("reward_margin must be finite and nonnegative")
    if required_violations is None:
        names = tuple(sorted(set(stable.violations) | set(candidate.violations)))
    else:
        names = tuple(dict.fromkeys(str(name) for name in required_violations))
        missing = [
            name
            for name in names
            if name not in stable.violations or name not in candidate.violations
        ]
        if missing:
            raise ValueError(
                "Candidate validation is missing required violations: "
                + ", ".join(missing)
            )
    tolerances = dict(violation_tolerances or {})
    reward_gain = float(candidate.reward) - float(stable.reward)
    if not math.isfinite(reward_gain):
        raise ValueError("Candidate rewards must be finite")
    deltas = {}
    failed_constraints = []
    for name in names:
        stable_value = float(stable.violations[name])
        candidate_value = float(candidate.violations[name])
        tolerance = float(tolerances.get(name, 0.0))
        if any(not math.isfinite(value) for value in (stable_value, candidate_value, tolerance)):
            raise ValueError("Candidate validation values must be finite")
        if tolerance < 0.0:
            raise ValueError("Violation tolerances must be nonnegative")
        delta = candidate_value - stable_value
        deltas[name] = delta
        if delta > tolerance:
            failed_constraints.append(name)
    # A zero margin still requires strict improvement; a positive margin may
    # be met exactly.
    reward_passed = (
        reward_gain > 0.0
        and reward_gain >= float(reward_margin)
    )
    accepted = reward_passed and not failed_constraints
    if not reward_passed:
        reason = "reward_margin"
    elif failed_constraints:
        reason = "constraint:" + ",".join(failed_constraints)
    else:
        reason = "accepted"
    return CandidateValidationResult(accepted, reward_gain, deltas, reason)


class DGLSCore:
    """Stateful, model-agnostic computation kernel for DGLS.

    This class implements spectral drift, hysteresis, budgeted layer selection,
    and acceptance rules.  Environment replay and atomic model-state commits
    remain the responsibility of the caller.
    """

    def __init__(
        self,
        *,
        threshold_on: float,
        threshold_off: float,
        budget_ratio: float,
        weight_activation: float = 0.45,
        weight_gradient: float = 0.45,
        weight_importance: float = 0.10,
    ) -> None:
        if threshold_on <= threshold_off or threshold_off < 0.0:
            raise ValueError("threshold_on must exceed nonnegative threshold_off")
        self.threshold_on = float(threshold_on)
        self.threshold_off = float(threshold_off)
        self.budget_ratio = float(budget_ratio)
        self.weight_activation = float(weight_activation)
        self.weight_gradient = float(weight_gradient)
        self.weight_importance = float(weight_importance)
        self.active = False
        self.previous_score = 0.0

    def update_drift(
        self,
        recent_features: torch.Tensor,
        reference_features: torch.Tensor,
        recent_gate_weights: torch.Tensor,
        bandwidth: float | Sequence[float] | None = None,
    ) -> tuple[float, bool, torch.Tensor, torch.Tensor]:
        score, band_drifts, band_weights = dispatch_weighted_spectral_drift(
            recent_features,
            reference_features,
            recent_gate_weights,
            bandwidth,
        )
        value = float(score.detach().cpu().item())
        self.active = hysteresis_state(
            value,
            self.active,
            self.threshold_on,
            self.threshold_off,
        )
        self.previous_score = value
        return value, self.active, band_drifts, band_weights

    def select_layers(
        self,
        activation: Mapping[str, float],
        gradient: Mapping[str, float],
        importance: Mapping[str, float],
        costs: Mapping[str, float],
    ) -> LayerSelection:
        return select_layers_under_budget(
            activation,
            gradient,
            importance,
            costs,
            self.budget_ratio,
            weight_activation=self.weight_activation,
            weight_gradient=self.weight_gradient,
            weight_importance=self.weight_importance,
        )
