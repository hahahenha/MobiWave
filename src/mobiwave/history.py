from __future__ import annotations

import numpy as np


class CausalDemandHistory:
    """Moving-window demand signal built only from requests observed so far."""

    name = "causal_history"

    def __init__(self, config, network) -> None:
        self.config = config
        self.network = network
        self.history: list[np.ndarray] = []

    def observe(self, t: int, requests, *, learn: bool = False) -> None:
        del learn
        while len(self.history) < t:
            self.history.append(
                np.zeros(self.network.zone_count, dtype=np.float32)
            )
        counts = np.zeros(self.network.zone_count, dtype=np.float32)
        for request in requests:
            counts[request.origin] += 1.0
        if len(self.history) == t:
            self.history.append(counts)
        else:
            self.history[t] = counts

    def expected_counts(self, t: int, window: int | None = None) -> np.ndarray:
        del t
        forecast_window = window or self.config.demand_window
        if not self.history:
            return np.zeros(self.network.zone_count, dtype=np.float32)
        lookback = min(
            len(self.history),
            max(1, self.config.demand_history_window),
        )
        recent = np.sum(self.history[-lookback:], axis=0)
        return recent / max(1, lookback) * forecast_window
