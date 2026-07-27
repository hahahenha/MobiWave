from __future__ import annotations

from typing import Any
import random

import numpy as np
import torch

from .config import SimulationConfig
from .environment import DispatchEnv
from .network import GridNetwork
from .policy import DGLSMobiWavePolicy
from .scenarios import DriftScenario


def run_episode(
    config: SimulationConfig,
    policy: DGLSMobiWavePolicy,
    *,
    seed: int | None = None,
    scenario: DriftScenario | None = None,
    metric_start_time: int = 0,
) -> tuple[DispatchEnv, dict[str, Any]]:
    """Run an initialized MobiWave policy on one independent stream."""

    episode_config = config.with_updates(
        seed=config.seed if seed is None else int(seed)
    )
    env = DispatchEnv(
        episode_config,
        scenario=scenario,
        metric_start_time=metric_start_time,
    )
    env.run(policy)
    return env, env.summary()


def train_mobiwave_offline(
    config: SimulationConfig,
    *,
    seed: int | None = None,
) -> tuple[DGLSMobiWavePolicy, DispatchEnv, dict[str, Any]]:
    """Fit MobiWave once on a causal no-drift history and initialize DGLS."""

    run_seed = config.seed if seed is None else int(seed)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    policy = DGLSMobiWavePolicy(
        config=config,
        network=GridNetwork(config.grid_rows, config.grid_cols),
    )
    history_horizon = max(
        int(config.horizon),
        policy.feature_builder.minimum_history
        + int(config.dgls_min_reference)
        + policy.model.forecast_horizon,
    )
    history_seed = run_seed + 100_000
    history_config = config.with_updates(
        horizon=history_horizon,
        seed=history_seed,
    )
    history_scenario = DriftScenario.for_horizon(
        "no_drift",
        history_horizon,
        onset=max(1, history_horizon // 3),
        recovery=max(2, 2 * history_horizon // 3),
        evaluation_control=True,
    )
    policy.begin_offline_training()
    history_env, history_summary = run_episode(
        history_config,
        policy,
        seed=history_seed,
        scenario=history_scenario,
    )
    policy.finalize_offline_training()
    return policy, history_env, history_summary
