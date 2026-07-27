# MobiWave

Minimal open-source implementation of:

> **MobiWave: Dispatch-Oriented Graph Wavelets and Drift-Guided Selective
> Optimization for Autonomous Fleet Rebalancing**

This directory contains only the MobiWave execution path:

- causal multi-horizon dispatch features;
- Chebyshev graph-wavelet bands and dispatch-aware gating;
- demand, edge-return, and feasible integer-flow heads;
- offline initialization;
- Dispatch-weighted Spectral Drift and budgeted layer selection;
- fast/slow online optimization with paired candidate validation;
- a small zone-level simulator and drift scenarios for reproducibility.

Policy baselines, alternative demand predictors, ablation runners, paper table
generation, dashboards, and visualization code from the research workspace are
intentionally excluded.

## Install

Python 3.10 or newer is required.

```bash
conda activate vehicle-dispatch-rl
cd public
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick start

The example first initializes MobiWave on an independent no-drift history, then
runs the accepted model with DGLS on a new test stream.

```bash
python examples/run_demo.py
python examples/run_demo.py --scenario sudden --output output/sudden
```

Use another configuration as follows:

```bash
python examples/run_demo.py --config configs/default.json
```

The output directory contains:

- `config.json`: the resolved configuration;
- `offline_summary.json`: statistics for the independent historical stream;
- `summary.json`: test-stream metrics;
- `logs.json`: step, vehicle, zone, scenario, and DGLS event logs;
- `mobiwave.pt`: the final MobiWave/DGLS checkpoint.

## Python API

```python
from mobiwave import (
    DriftScenario,
    SimulationConfig,
    run_episode,
    train_mobiwave_offline,
)

config = SimulationConfig(horizon=80, seed=2026)
policy, _, _ = train_mobiwave_offline(config)
scenario = DriftScenario.for_horizon("sudden", config.horizon)
env, summary = run_episode(
    config,
    policy,
    seed=config.seed + 99,
    scenario=scenario,
)
print(summary)
```

`MobiWavePolicy` exposes the backbone without online adaptation.
`DGLSMobiWavePolicy` is the complete offline-initialized and drift-adaptive
policy used by `train_mobiwave_offline`.

## Source layout

```text
src/mobiwave/
├── model.py          # graph-wavelet backbone and feasible dispatch
├── policy.py         # offline initialization and online DGLS
├── graph_wavelet.py  # Chebyshev bands and dispatch-aware gate
├── adaptation/       # drift, transactions, replay, accepted state
├── optimizer.py      # DGLS fast/slow optimizer
├── environment.py    # reproducible zone-level simulator
├── config.py         # MobiWave-only configuration
└── training.py       # minimal public training/evaluation API
```

## Reproducibility boundary

The simulator is self-contained and does not ship proprietary trip data. It is
intended for executable examples and controlled drift studies. Results on a
real fleet require users to provide their own causal snapshots with the fields
consumed by `MobiWavePolicy.decide`.

## License

MIT. See [LICENSE](LICENSE).
