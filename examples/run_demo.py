from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from mobiwave import (
    DriftScenario,
    load_config,
    run_episode,
    save_config,
    train_mobiwave_offline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone MobiWave demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "default.json",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "no_drift",
            "sudden",
            "gradual",
            "structural",
            "recurring",
            "supply_side",
        ),
        default="sudden",
    )
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    policy, _, offline_summary = train_mobiwave_offline(
        config,
        seed=config.seed,
    )
    scenario = DriftScenario.for_horizon(
        args.scenario,
        config.horizon,
        intensity=args.intensity,
    )
    env, summary = run_episode(
        config,
        policy,
        seed=config.seed + 99,
        scenario=scenario,
    )

    save_config(config, args.output / "config.json")
    write_json(args.output / "offline_summary.json", offline_summary)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "logs.json", env.logs_as_dict())
    torch.save(policy.export_state(), args.output / "mobiwave.pt")
    print(json.dumps(summary, indent=2, default=json_default))
    print(f"Artifacts written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
