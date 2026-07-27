from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any
import json
import math

from .reward import (
    DEFAULT_REBALANCE_REWARD_COMPONENTS,
    normalize_reward_components,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the MobiWave simulator and adaptation pipeline."""

    # Environment.
    grid_rows: int = 20
    grid_cols: int = 20
    fleet_size: int = 50
    horizon: int = 180_000
    seed: int = 42
    max_wait: int = 15
    demand_rate: float = 1.45
    demand_window: int = 10
    demand_history_window: int = 30
    state_hop_radius: int = 3

    # Dispatch-oriented graph wavelets.
    graph_wavelet_heat_scales: tuple[float, ...] = (4.0, 2.0, 1.0)
    graph_wavelet_chebyshev_order: int = 5
    mobiwave_temporal_horizons: tuple[int, ...] = (2, 4, 8)
    mobiwave_forecast_horizon: int = 3
    mobiwave_hidden_dim: int = 64
    mobiwave_zone_embedding_dim: int = 0
    mobiwave_steps_per_hour: float = 1.0
    mobiwave_service_capacity: float = 1.0
    mobiwave_scale_dropout: float = 0.10
    mobiwave_demand_loss_weight: float = 1.0
    mobiwave_return_loss_weight: float = 0.20
    mobiwave_policy_loss_weight: float = 1.0
    mobiwave_gate_balance_weight: float = 0.01
    mobiwave_pretrain_epochs: int = 500
    mobiwave_pretrain_batch_size: int = 32
    mobiwave_use_temporal_summaries: bool = True
    mobiwave_use_gating: bool = True
    mobiwave_filter_mode: str = "graph_wavelet"
    policy_discount: float = 0.92
    policy_clip: float = 0.20

    # Drift-Guided Layer-Selective Optimization (DGLS).
    dgls_train_window: int = 12
    dgls_validation_window: int = 6
    dgls_reference_capacity: int = 512
    dgls_min_reference: int = 12
    dgls_time_period: int = 24
    dgls_time_bins: int = 6
    dgls_gap_bin_width: float = 1.0
    dgls_threshold_on: float = 0.20
    dgls_threshold_off: float = 0.025
    dgls_bandwidth_floor: float = 1e-3
    dgls_mmd_sample_cap: int = 256
    dgls_bandwidth_sample_cap: int = 512
    dgls_budget_ratio: float = 0.10
    dgls_activation_score_weight: float = 0.45
    dgls_gradient_score_weight: float = 0.45
    dgls_importance_score_weight: float = 0.10
    dgls_base_lr: float = 3e-4
    dgls_eta_min: float = 0.10
    dgls_inner_steps: int = 4
    dgls_beta_persistence: float = 0.90
    dgls_kappa_shock: float = 1.0
    dgls_kappa_persistence: float = 1.0
    dgls_slow_interval_min: int = 2
    dgls_slow_interval_max: int = 32
    dgls_importance_beta: float = 0.99
    dgls_reference_loss_weight: float = 0.50
    dgls_activation_loss_weight: float = 0.10
    dgls_importance_loss_weight: float = 1.0
    dgls_reward_margin: float = 0.0
    dgls_wait_tolerance: float = 0.0
    dgls_infeasible_tolerance: float = 0.0
    dgls_service_shortfall_tolerance: float = 0.0
    dgls_attempt_interval: int = 1

    # Fast/slow optimizer memories used by DGLS.
    m3_beta_fast: float = 0.9
    m3_beta_slow: float = 0.99
    m3_beta_second: float = 0.999
    m3_slow_update_interval: int = 8
    m3_slow_weight: float = 0.35
    m3_muon_steps: int = 4

    # Simulator reward and accounting.
    reward_components: tuple[str, ...] = DEFAULT_REBALANCE_REWARD_COMPONENTS
    vehicle_move_cost_per_grid: float = 1.0
    passenger_trip_revenue_per_grid: float | None = None
    pickup_penalty: float = 0.35
    travel_penalty: float = 0.08
    cancel_penalty: float = 65.0
    rebalance_penalty: float = 0.08
    idle_penalty: float = 0.025
    stay_streak_penalty: float = 0.003
    stay_streak_growth: float = 0.18
    stay_streak_penalty_cap: float = 3.0
    idle_cluster_penalty: float = 0.04
    idle_cluster_free_threshold: int = 2
    rebalance_radius: int = 2
    rebalance_deficit_weight: float = 0.45
    rebalance_improvement_weight: float = 0.32
    rebalance_worsen_penalty: float = 0.12
    rebalance_open_request_weight: float = 0.20
    rebalance_move_threshold: float = 0.15
    rebalance_move_fixed_cost: float = 0.08
    rebalance_target_idle_penalty: float = 0.02
    rebalance_idle_spread_weight: float = 0.18
    invalid_action_penalty: float = 35.0

    def __post_init__(self) -> None:
        positive_integers = {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "horizon": self.horizon,
            "demand_window": self.demand_window,
            "demand_history_window": self.demand_history_window,
            "mobiwave_forecast_horizon": self.mobiwave_forecast_horizon,
            "mobiwave_hidden_dim": self.mobiwave_hidden_dim,
            "mobiwave_pretrain_epochs": self.mobiwave_pretrain_epochs,
            "mobiwave_pretrain_batch_size": self.mobiwave_pretrain_batch_size,
            "dgls_train_window": self.dgls_train_window,
            "dgls_validation_window": self.dgls_validation_window,
            "dgls_reference_capacity": self.dgls_reference_capacity,
            "dgls_min_reference": self.dgls_min_reference,
            "dgls_time_period": self.dgls_time_period,
            "dgls_time_bins": self.dgls_time_bins,
            "dgls_mmd_sample_cap": self.dgls_mmd_sample_cap,
            "dgls_bandwidth_sample_cap": self.dgls_bandwidth_sample_cap,
            "dgls_inner_steps": self.dgls_inner_steps,
            "dgls_slow_interval_min": self.dgls_slow_interval_min,
            "dgls_slow_interval_max": self.dgls_slow_interval_max,
            "dgls_attempt_interval": self.dgls_attempt_interval,
            "m3_slow_update_interval": self.m3_slow_update_interval,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        nonnegative_integers = {
            "fleet_size": self.fleet_size,
            "max_wait": self.max_wait,
            "state_hop_radius": self.state_hop_radius,
            "mobiwave_zone_embedding_dim": self.mobiwave_zone_embedding_dim,
            "m3_muon_steps": self.m3_muon_steps,
            "idle_cluster_free_threshold": self.idle_cluster_free_threshold,
            "rebalance_radius": self.rebalance_radius,
        }
        for name, value in nonnegative_integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

        horizons = tuple(sorted({int(value) for value in self.mobiwave_temporal_horizons}))
        if not horizons or any(value < 1 for value in horizons):
            raise ValueError("mobiwave_temporal_horizons must contain positive integers")
        object.__setattr__(self, "mobiwave_temporal_horizons", horizons)

        heat_scales = tuple(float(value) for value in self.graph_wavelet_heat_scales)
        if (
            not heat_scales
            or any(not math.isfinite(value) or value <= 0.0 for value in heat_scales)
            or any(left <= right for left, right in zip(heat_scales, heat_scales[1:]))
        ):
            raise ValueError(
                "graph_wavelet_heat_scales must be finite, positive, and strictly decreasing"
            )
        object.__setattr__(self, "graph_wavelet_heat_scales", heat_scales)
        if not 1 <= int(self.graph_wavelet_chebyshev_order) <= 32:
            raise ValueError("graph_wavelet_chebyshev_order must be between 1 and 32")

        if self.dgls_min_reference > self.dgls_reference_capacity:
            raise ValueError("dgls_min_reference cannot exceed dgls_reference_capacity")
        if self.dgls_slow_interval_min > self.dgls_slow_interval_max:
            raise ValueError("dgls_slow_interval_min cannot exceed dgls_slow_interval_max")
        if not self.dgls_threshold_on > self.dgls_threshold_off >= 0.0:
            raise ValueError("DGLS thresholds must satisfy threshold_on > threshold_off >= 0")
        if not 0.0 < self.dgls_budget_ratio <= 1.0:
            raise ValueError("dgls_budget_ratio must be in (0, 1]")
        if not 0.0 < self.dgls_eta_min < 1.0:
            raise ValueError("dgls_eta_min must be in (0, 1)")
        if not 0.0 <= self.mobiwave_scale_dropout < 1.0:
            raise ValueError("mobiwave_scale_dropout must be in [0, 1)")
        if not 0.0 <= self.policy_discount <= 1.0:
            raise ValueError("policy_discount must be in [0, 1]")
        if not 0.0 <= self.policy_clip < 1.0:
            raise ValueError("policy_clip must be in [0, 1)")

        filter_mode = str(self.mobiwave_filter_mode).strip().lower().replace("-", "_")
        if filter_mode not in {"graph_wavelet", "gcn"}:
            raise ValueError("mobiwave_filter_mode must be graph_wavelet or gcn")
        object.__setattr__(self, "mobiwave_filter_mode", filter_mode)
        object.__setattr__(
            self,
            "reward_components",
            normalize_reward_components(self.reward_components),
        )
        if self.passenger_trip_revenue_per_grid is None:
            object.__setattr__(
                self,
                "passenger_trip_revenue_per_grid",
                3.0 * self.vehicle_move_cost_per_grid,
            )

        nonnegative_reals = (
            self.demand_rate,
            self.mobiwave_demand_loss_weight,
            self.mobiwave_return_loss_weight,
            self.mobiwave_policy_loss_weight,
            self.mobiwave_gate_balance_weight,
            self.dgls_activation_score_weight,
            self.dgls_gradient_score_weight,
            self.dgls_importance_score_weight,
            self.dgls_kappa_shock,
            self.dgls_kappa_persistence,
            self.dgls_reference_loss_weight,
            self.dgls_activation_loss_weight,
            self.dgls_importance_loss_weight,
            self.dgls_reward_margin,
            self.dgls_wait_tolerance,
            self.dgls_infeasible_tolerance,
            self.dgls_service_shortfall_tolerance,
            self.vehicle_move_cost_per_grid,
            self.passenger_trip_revenue_per_grid,
            self.pickup_penalty,
            self.travel_penalty,
            self.cancel_penalty,
            self.rebalance_penalty,
            self.idle_penalty,
            self.stay_streak_penalty,
            self.stay_streak_growth,
            self.stay_streak_penalty_cap,
            self.idle_cluster_penalty,
            self.rebalance_deficit_weight,
            self.rebalance_improvement_weight,
            self.rebalance_worsen_penalty,
            self.rebalance_open_request_weight,
            self.rebalance_move_threshold,
            self.rebalance_move_fixed_cost,
            self.rebalance_target_idle_penalty,
            self.rebalance_idle_spread_weight,
            self.invalid_action_penalty,
        )
        if any(
            not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in nonnegative_reals
        ):
            raise ValueError("loss weights, tolerances, and reward weights must be nonnegative")
        positive_reals = (
            self.mobiwave_steps_per_hour,
            self.mobiwave_service_capacity,
            self.dgls_gap_bin_width,
            self.dgls_bandwidth_floor,
            self.dgls_base_lr,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive_reals):
            raise ValueError("time, capacity, bandwidth, and learning-rate values must be positive")
        if any(not 0.0 <= float(beta) < 1.0 for beta in (
            self.m3_beta_fast,
            self.m3_beta_slow,
            self.m3_beta_second,
            self.dgls_beta_persistence,
            self.dgls_importance_beta,
        )):
            raise ValueError("optimizer and persistence beta values must be in [0, 1)")
        if not 0.0 <= self.m3_slow_weight <= 1.0:
            raise ValueError("m3_slow_weight must be in [0, 1]")

    @property
    def zone_count(self) -> int:
        return self.grid_rows * self.grid_cols

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **kwargs: Any) -> "SimulationConfig":
        return replace(self, **kwargs)


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid_keys = {field.name for field in fields(SimulationConfig)}
    return SimulationConfig(**{key: value for key, value in data.items() if key in valid_keys})


def save_config(config: SimulationConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
