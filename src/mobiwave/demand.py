from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math
import numpy as np

from .config import SimulationConfig
from .entities import Request
from .network import GridNetwork
from .scenarios import DriftScenario


@dataclass(frozen=True)
class Hotspot:
    row: float
    col: float
    sigma: float
    phase: float
    strength: float


class DemandGenerator:
    """Time-varying zone demand with commute-like spatial drift."""

    def __init__(
        self,
        config: SimulationConfig,
        network: GridNetwork,
        rng: np.random.Generator,
        scenario: DriftScenario | None = None,
    ) -> None:
        self.config = config
        self.network = network
        self.rng = rng
        self.scenario = scenario or DriftScenario.for_horizon("no_drift", config.horizon)
        self.hotspots = [
            Hotspot(1.0, 1.1, 1.15, 0.00, 1.45),
            Hotspot(network.rows - 2.0, network.cols - 2.0, 1.25, 0.34, 1.35),
            Hotspot((network.rows - 1) / 2, (network.cols - 1) / 2, 1.50, 0.68, 1.75),
        ]

    def zone_weights(self, t: int) -> np.ndarray:
        return self._zone_weights_at(pattern_t=t, scenario_t=t)

    def _zone_weights_at(self, pattern_t: int, scenario_t: int) -> np.ndarray:
        weights = self._base_zone_weights(pattern_t)
        strength = self.scenario.strength(scenario_t)
        if strength <= 0.0:
            return weights

        if self.scenario.family == "gradual":
            shifted = self._shifted_commute_weights(pattern_t)
            blend = min(1.0, max(0.0, strength))
            weights = (1.0 - blend) * weights + blend * shifted
        elif self.scenario.family in {"sudden", "recurring"}:
            event = self._event_hotspot_weights()
            scale = (1.25 if self.scenario.family == "sudden" else 0.90) * strength
            weights = weights + scale * event

        return self._normalize(weights)

    def _base_zone_weights(self, t: int) -> np.ndarray:
        weights = np.full(self.network.zone_count, 0.08, dtype=float)
        progress = (t % max(self.config.horizon, 1)) / max(self.config.horizon, 1)
        for hotspot in self.hotspots:
            wave = 0.62 + 0.38 * math.cos(2 * math.pi * (progress - hotspot.phase))
            for zone in range(self.network.zone_count):
                row, col = self.network.to_rc(zone)
                dist2 = (row - hotspot.row) ** 2 + (col - hotspot.col) ** 2
                weights[zone] += hotspot.strength * wave * math.exp(-dist2 / (2 * hotspot.sigma**2))
        return self._normalize(weights)

    def _shifted_commute_weights(self, t: int) -> np.ndarray:
        """Return the alternative commute pattern used by gradual drift."""

        weights = np.full(self.network.zone_count, 0.08, dtype=float)
        progress = (t % max(self.config.horizon, 1)) / max(self.config.horizon, 1)
        max_row = max(0, self.network.rows - 1)
        max_col = max(0, self.network.cols - 1)
        for hotspot in self.hotspots:
            wave = 0.62 + 0.38 * math.cos(2 * math.pi * (progress - hotspot.phase))
            shifted_row = max_row - hotspot.row
            shifted_col = max_col - hotspot.col
            for zone in range(self.network.zone_count):
                row, col = self.network.to_rc(zone)
                dist2 = (row - shifted_row) ** 2 + (col - shifted_col) ** 2
                weights[zone] += hotspot.strength * wave * math.exp(-dist2 / (2 * hotspot.sigma**2))
        return self._normalize(weights)

    def _event_hotspot_weights(self) -> np.ndarray:
        center_row = 0.35 * max(0, self.network.rows - 1)
        center_col = 0.70 * max(0, self.network.cols - 1)
        sigma = max(0.75, 0.15 * min(self.network.rows, self.network.cols))
        weights = np.zeros(self.network.zone_count, dtype=float)
        for zone in range(self.network.zone_count):
            row, col = self.network.to_rc(zone)
            dist2 = (row - center_row) ** 2 + (col - center_col) ** 2
            weights[zone] = math.exp(-dist2 / (2 * sigma**2))
        return self._normalize(weights)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        normalized = np.asarray(weights, dtype=float).copy()
        normalized[~np.isfinite(normalized)] = 0.0
        np.maximum(normalized, 0.0, out=normalized)
        total = float(normalized.sum())
        if total <= 0.0:
            return np.full(self.network.zone_count, 1.0 / max(1, self.network.zone_count), dtype=float)
        normalized /= total
        return normalized

    def expected_counts(self, t: int, window: int | None = None) -> np.ndarray:
        window = window or self.config.demand_window
        counts = np.zeros(self.network.zone_count, dtype=float)
        for dt in range(window):
            rate = self.rate_at(t + dt)
            counts += rate * self.zone_weights(t + dt)
        return counts

    def rate_at(self, t: int) -> float:
        cycle = 0.72 + 0.28 * math.sin(2 * math.pi * ((t % 90) / 90.0 - 0.12))
        pulse = 0.24 if 45 <= (t % 120) <= 75 else 0.0
        base_rate = max(0.15, self.config.demand_rate * (cycle + pulse))
        return base_rate * self.scenario.demand_rate_multiplier(t)

    def sample_requests(self, t: int) -> List[Request]:
        # Independent substreams keep count, locations, destinations, and fares
        # aligned between a drift stream and its matched control wherever their
        # respective distributions are unchanged.
        count_rng = self._step_rng(t, stream=0)
        count = int(count_rng.poisson(self.rate_at(t)))
        if count <= 0:
            return []
        weights = self.zone_weights(t)
        origins = self._step_rng(t, stream=1).choice(
            self.network.zone_count,
            size=count,
            p=weights,
        )
        requests = []
        for request_index, origin in enumerate(origins):
            destination = self._sample_destination(
                int(origin),
                t,
                self._step_rng(t, stream=2, item=request_index),
            )
            fare = 4.5 + float(
                self._step_rng(t, stream=3, item=request_index).uniform(0.0, 2.0)
            )
            requests.append(
                Request(
                    # A time-local ID prevents an active drift from shifting all
                    # request IDs in the matched post-drift replay.
                    request_id=(int(t) << 32) + request_index,
                    origin=int(origin),
                    destination=destination,
                    created_at=t,
                    fare=fare,
                )
            )
        return requests

    def _sample_destination(
        self,
        origin: int,
        t: int,
        rng: np.random.Generator | None = None,
    ) -> int:
        if self.network.zone_count <= 1:
            return origin
        rng = rng if rng is not None else self._step_rng(t, stream=2)
        # The base commute cycle looks ahead as before, while drift is applied
        # according to the request's creation step so the evaluator's onset and
        # recovery boundaries remain exact.
        weights = self._zone_weights_at(
            pattern_t=t + self.config.demand_window,
            scenario_t=t,
        )
        away = np.zeros(self.network.zone_count, dtype=float)
        for zone in range(self.network.zone_count):
            away[zone] = 0.25 + self.network.hex_distance(origin, zone)
        weights = weights * away
        weights[origin] = 0.0
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            return origin
        weights /= total
        return int(rng.choice(self.network.zone_count, p=weights))

    def _step_rng(self, t: int, *, stream: int = 0, item: int = 0) -> np.random.Generator:
        """Create a matched random stream whose state depends only on seed and step."""

        seed_sequence = np.random.SeedSequence(
            [int(self.config.seed), int(t), int(stream), int(item), 0x44454D41]
        )
        return np.random.default_rng(seed_sequence)
