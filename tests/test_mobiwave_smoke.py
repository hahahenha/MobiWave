from __future__ import annotations

import numpy as np
import torch

from mobiwave import (
    DriftScenario,
    MobiWaveDispatchNet,
    SimulationConfig,
    run_episode,
    train_mobiwave_offline,
)
from mobiwave.network import GridNetwork


def tiny_config(**updates) -> SimulationConfig:
    values = {
        "grid_rows": 2,
        "grid_cols": 2,
        "fleet_size": 4,
        "horizon": 8,
        "demand_rate": 1.0,
        "demand_window": 1,
        "demand_history_window": 4,
        "mobiwave_temporal_horizons": (1,),
        "mobiwave_forecast_horizon": 1,
        "mobiwave_hidden_dim": 8,
        "mobiwave_zone_embedding_dim": 2,
        "mobiwave_pretrain_epochs": 1,
        "mobiwave_pretrain_batch_size": 4,
        "graph_wavelet_heat_scales": (2.0, 1.0),
        "graph_wavelet_chebyshev_order": 2,
        "dgls_min_reference": 2,
        "dgls_reference_capacity": 8,
        "dgls_train_window": 2,
        "dgls_validation_window": 2,
        "dgls_inner_steps": 1,
        "dgls_threshold_on": 1e6,
        "dgls_threshold_off": 0.0,
        "dgls_budget_ratio": 1.0,
        "m3_muon_steps": 0,
    }
    values.update(updates)
    return SimulationConfig(**values)


def test_dispatch_network_masks_edges_and_conserves_supply() -> None:
    torch.manual_seed(7)
    network = GridNetwork(2, 2)
    model = MobiWaveDispatchNet(
        network=network,
        input_dim=12,
        hidden_dim=8,
        heat_scales=(2.0, 1.0),
        chebyshev_order=2,
        forecast_horizon=1,
    )
    supply = torch.tensor([[2, 1, 0, 1]])
    outputs = model(torch.randn(1, network.zone_count, 12), available_supply=supply)
    allocation = model.allocate(outputs["edge_probabilities"], supply)

    torch.testing.assert_close(model.dense_flow(allocation).sum(dim=-1), supply)
    assert torch.all(model.dense_flow(allocation)[:, ~model.feasible_mask] == 0)
    np.testing.assert_allclose(
        outputs["gate_weights"].detach().numpy().sum(axis=-1),
        1.0,
        atol=1e-6,
    )


def test_offline_initialization_and_independent_test_stream() -> None:
    config = tiny_config(dgls_threshold_on=1e-6)
    policy, offline_env, _ = train_mobiwave_offline(config, seed=7)
    scenario = DriftScenario.for_horizon(
        "sudden",
        config.horizon,
        onset=3,
        recovery=7,
    )
    env, summary = run_episode(config, policy, seed=106, scenario=scenario)
    runtime_events = [
        event
        for event in policy.dgls_events
        if event["event_type"] == "adaptation_runtime"
    ]

    assert offline_env.config.seed != env.config.seed
    assert policy.controller is not None
    assert runtime_events
    assert all(event["paired_replay_available"] for event in runtime_events)
    assert summary["generated"] >= summary["served"]
    assert len(env.summary_log) == config.horizon
