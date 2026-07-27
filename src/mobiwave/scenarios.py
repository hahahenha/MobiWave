from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict
import hashlib
import math


DRIFT_FAMILIES = (
    "no_drift",
    "sudden",
    "gradual",
    "structural",
    "recurring",
    "supply_side",
)

DRIFT_LABELS = {
    "no_drift": "Matched no-drift control",
    "sudden": "Sudden local shock",
    "gradual": "Gradual mobility shift",
    "structural": "Structural corridor change",
    "recurring": "Recurring event pattern",
    "supply_side": "Supply-side downtime",
}

DRIFT_DESCRIPTIONS = {
    "no_drift": "Keeps the base request, road, and fleet process unchanged over the matched window.",
    "sudden": "Injects an abrupt local demand surge and travel-time shock, such as an accident or large event.",
    "gradual": "Moves demand smoothly from the historical pattern to a shifted commuting pattern and back.",
    "structural": "Persistently increases travel time across a central road corridor during the drift window.",
    "recurring": "Introduces an event hotspot, removes it, and later brings the same pattern back.",
    "supply_side": "Temporarily makes a deterministic subset of vehicles unavailable, mimicking charging or downtime.",
}

_FAMILY_ALIASES = {
    "none": "no_drift",
    "control": "no_drift",
    "no-drift": "no_drift",
    "supply": "supply_side",
    "supply-side": "supply_side",
}


def normalize_drift_family(value: str) -> str:
    family = str(value).strip().lower().replace(" ", "_")
    family = _FAMILY_ALIASES.get(family, family)
    if family not in DRIFT_FAMILIES:
        raise ValueError(f"Unknown drift family: {value!r}")
    return family


@dataclass(frozen=True)
class DriftScenario:
    """Private evaluation schedule for one mobility-drift stream.

    The object is intentionally kept outside ``SimulationConfig`` because the
    latter is included in policy snapshots. Ground-truth onset and recovery
    times must be available to the evaluator, not to the dispatch policy.
    """

    family: str
    onset: int
    recovery: int
    intensity: float = 1.0
    matched_to: str = ""
    evaluation_control: bool = False

    def __post_init__(self) -> None:
        family = normalize_drift_family(self.family)
        object.__setattr__(self, "family", family)
        if self.onset < 0:
            raise ValueError("scenario onset must be non-negative")
        if self.recovery <= self.onset:
            raise ValueError("scenario recovery must be greater than onset")
        if not math.isfinite(self.intensity) or self.intensity < 0:
            raise ValueError("scenario intensity must be a finite non-negative value")
        if self.matched_to:
            object.__setattr__(self, "matched_to", normalize_drift_family(self.matched_to))
        if self.evaluation_control and self.family != "no_drift":
            raise ValueError("only a no-drift scenario can be an evaluation control")

    @classmethod
    def for_horizon(
        cls,
        family: str,
        horizon: int,
        *,
        onset: int | None = None,
        recovery: int | None = None,
        intensity: float = 1.0,
        matched_to: str = "",
        evaluation_control: bool = False,
    ) -> "DriftScenario":
        horizon = max(1, int(horizon))
        default_onset = max(0, min(horizon - 1, int(round(0.35 * horizon))))
        default_recovery = max(default_onset + 1, min(horizon, int(round(0.75 * horizon))))
        onset_value = default_onset if onset is None else max(0, min(horizon - 1, int(onset)))
        recovery_value = default_recovery if recovery is None else max(onset_value + 1, min(horizon, int(recovery)))
        return cls(
            family=family,
            onset=onset_value,
            recovery=recovery_value,
            intensity=float(intensity),
            matched_to=matched_to,
            evaluation_control=evaluation_control,
        )

    @property
    def label(self) -> str:
        return DRIFT_LABELS[self.family]

    @property
    def description(self) -> str:
        return DRIFT_DESCRIPTIONS[self.family]

    @property
    def recurrence_step(self) -> int:
        duration = max(1, self.recovery - self.onset)
        return min(self.recovery - 1, self.onset + max(1, (2 * duration) // 3))

    @property
    def first_event_end(self) -> int:
        duration = max(1, self.recovery - self.onset)
        return min(self.recovery, self.onset + max(1, duration // 3))

    def window_phase(self, t: int) -> str:
        if t < self.onset:
            return "pre"
        if t >= self.recovery:
            return "post"
        return "active"

    def phase(self, t: int) -> str:
        window_phase = self.window_phase(t)
        if window_phase != "active":
            return window_phase
        if self.family == "no_drift":
            return "control"
        if self.family == "recurring":
            if t < self.first_event_end:
                return "initial_event"
            if t < self.recurrence_step:
                return "event_gap"
            return "recurrence"
        return "active"

    def strength(self, t: int) -> float:
        if self.family == "no_drift" or t < self.onset or t >= self.recovery:
            return 0.0
        if self.family == "gradual":
            duration = max(1, self.recovery - self.onset)
            midpoint = self.onset + duration / 2.0
            if t <= midpoint:
                fraction = (t - self.onset) / max(1.0, midpoint - self.onset)
            else:
                fraction = (self.recovery - t) / max(1.0, self.recovery - midpoint)
            return self.intensity * min(1.0, max(0.0, fraction))
        if self.family == "recurring":
            active = t < self.first_event_end or t >= self.recurrence_step
            return self.intensity if active else 0.0
        return self.intensity

    def demand_rate_multiplier(self, t: int) -> float:
        strength = self.strength(t)
        if self.family == "sudden":
            return 1.0 + 0.75 * strength
        if self.family == "gradual":
            return 1.0 + 0.20 * strength
        if self.family == "recurring":
            return 1.0 + 0.40 * strength
        return 1.0

    def travel_time_multiplier(self, network, origin: int, destination: int, t: int) -> float:
        strength = self.strength(t)
        if strength <= 0.0 or origin == destination:
            return 1.0
        if self.family == "sudden":
            if self._near_event_center(network, origin) or self._near_event_center(network, destination):
                return 1.0 + 1.25 * strength
        if self.family == "structural" and self._crosses_structural_corridor(network, origin, destination):
            return 1.0 + 1.75 * strength
        return 1.0

    def unavailable_fraction(self, t: int) -> float:
        if self.family != "supply_side":
            return 0.0
        return min(0.80, 0.35 * self.strength(t))

    def edge_available(self, network, origin: int, destination: int, t: int) -> bool:
        """Return the observed road connectivity at one decision step."""

        if origin == destination or self.family != "structural" or self.strength(t) <= 0.0:
            return True
        origin_row, origin_col = network.to_rc(origin)
        destination_row, destination_col = network.to_rc(destination)
        corridor_col = max(1, network.cols // 2)
        crosses = (origin_col < corridor_col <= destination_col) or (
            destination_col < corridor_col <= origin_col
        )
        central_band = (
            min(origin_row, destination_row)
            <= network.rows // 2
            <= max(origin_row, destination_row) + 1
        )
        # A deterministic subset of central crossing links closes while the
        # structural scenario is active.  The policy observes connectivity,
        # never the evaluator-only onset/recovery boundaries.
        return not (crosses and central_band)

    def graph_version(self, network, t: int) -> str:
        edges = tuple(
            (source, destination)
            for source in range(network.zone_count)
            for destination in sorted(
                {
                    int(network.move(source, action))
                    for action in network.valid_actions(source)
                }
            )
            if destination == source
            or self.edge_available(network, source, destination, t)
        )
        digest = hashlib.sha256(repr(edges).encode("utf-8")).hexdigest()[:16]
        return f"{network.rows}x{network.cols}:{digest}"

    def marker_at(self, t: int) -> str | None:
        if self.family == "no_drift" and not self.evaluation_control:
            return None
        if t == self.onset:
            return "control_window_start" if self.family == "no_drift" else "drift_onset"
        if self.family == "recurring" and t == self.recurrence_step:
            return "drift_recurrence"
        if t == self.recovery:
            return "control_window_end" if self.family == "no_drift" else "drift_recovery"
        return None

    def matched_control(self) -> "DriftScenario":
        return DriftScenario(
            family="no_drift",
            onset=self.onset,
            recovery=self.recovery,
            intensity=self.intensity,
            matched_to=self.family,
            evaluation_control=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "label": self.label,
                "description": self.description,
                "recurrence_step": self.recurrence_step if self.family == "recurring" else None,
            }
        )
        return payload

    @staticmethod
    def _near_event_center(network, zone: int) -> bool:
        row, col = network.to_rc(zone)
        center_row = 0.35 * max(0, network.rows - 1)
        center_col = 0.70 * max(0, network.cols - 1)
        radius = max(1.0, 0.18 * min(network.rows, network.cols))
        return (row - center_row) ** 2 + (col - center_col) ** 2 <= radius**2

    @staticmethod
    def _crosses_structural_corridor(network, origin: int, destination: int) -> bool:
        origin_row, origin_col = network.to_rc(origin)
        destination_row, destination_col = network.to_rc(destination)
        corridor_col = max(1, network.cols // 2)
        crosses = (origin_col < corridor_col <= destination_col) or (
            destination_col < corridor_col <= origin_col
        )
        near_corridor = abs(origin_col - corridor_col) <= 1 or abs(destination_col - corridor_col) <= 1
        central_rows = abs(origin_row - destination_row) <= max(2, network.rows // 3)
        return (crosses or near_corridor) and central_rows

