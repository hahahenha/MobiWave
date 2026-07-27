from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple
import math


ACTION_NAMES = [
    "stay",
    "east",
    "north_east",
    "north_west",
    "west",
    "south_west",
    "south_east",
]

EVEN_ROW_DELTAS = {
    0: (0, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (-1, -1),
    4: (0, -1),
    5: (1, -1),
    6: (1, 0),
}

ODD_ROW_DELTAS = {
    0: (0, 0),
    1: (0, 1),
    2: (-1, 1),
    3: (-1, 0),
    4: (0, -1),
    5: (1, 0),
    6: (1, 1),
}


@dataclass(frozen=True)
class GridNetwork:
    """Odd-row offset hex grid used as the zone-level road network."""

    rows: int
    cols: int

    @property
    def zone_count(self) -> int:
        return self.rows * self.cols

    def to_zone(self, row: int, col: int) -> int:
        return row * self.cols + col

    def to_rc(self, zone: int) -> Tuple[int, int]:
        return zone // self.cols, zone % self.cols

    def coord(self, zone: int) -> Tuple[float, float]:
        row, col = self.to_rc(zone)
        x = math.sqrt(3.0) * (col + 0.5 * (row % 2))
        y = 1.5 * (self.rows - 1 - row)
        return x, y

    def manhattan(self, a: int, b: int) -> int:
        return self.hex_distance(a, b)

    def hex_distance(self, a: int, b: int) -> int:
        return max(
            abs(left - right)
            for left, right in zip(self._cube(a), self._cube(b))
        )

    def move(self, zone: int, action: int) -> int:
        row, col = self.to_rc(zone)
        deltas = ODD_ROW_DELTAS if row % 2 else EVEN_ROW_DELTAS
        if action not in deltas:
            return zone
        dr, dc = deltas[action]
        nr = row + dr
        nc = col + dc
        if not self._in_bounds(nr, nc):
            return zone
        return self.to_zone(nr, nc)

    def valid_actions(self, zone: int) -> List[int]:
        return [action for action in range(len(ACTION_NAMES)) if self.move(zone, action) != zone or action == 0]

    def neighbors(self, zone: int, radius: int = 1) -> List[int]:
        return [
            candidate
            for candidate in range(self.zone_count)
            if self.hex_distance(zone, candidate) <= radius
        ]

    @lru_cache(maxsize=None)
    def static_direction_bias(self, a: int, b: int) -> float:
        if a == b:
            return 1.0
        ar, ac = self.to_rc(a)
        br, bc = self.to_rc(b)
        eastbound = 1.10 if bc > ac else 0.98
        cbd_row = (self.rows - 1) / 2.0
        cbd_col = (self.cols - 1) / 2.0
        toward_cbd = (
            abs(br - cbd_row) + abs(bc - cbd_col)
            < abs(ar - cbd_row) + abs(ac - cbd_col)
        )
        corridor = 1.18 if toward_cbd else 1.0
        ripple = 1.0 + 0.08 * math.sin((a * 17 + b * 31) % 13)
        return max(0.75, eastbound * corridor * ripple)

    def travel_time(self, a: int, b: int, t: int = 0) -> int:
        if a == b:
            return 1
        base = 1.0 + 1.8 * self.manhattan(a, b)
        rush = 1.0 + 0.18 * math.sin(2 * math.pi * (t % 60) / 60.0)
        return max(1, int(math.ceil(base * rush * self.static_direction_bias(a, b))))

    def free_flow_time(self, a: int, b: int) -> int:
        """Return the static no-rush time used by graph weights."""

        if a == b:
            return 1
        base = 1.0 + 1.8 * self.manhattan(a, b)
        return max(1, int(math.ceil(base * self.static_direction_bias(a, b))))

    def all_zone_coords(self) -> List[Dict[str, float]]:
        return [
            {"zone": zone, "x": self.coord(zone)[0], "y": self.coord(zone)[1]}
            for zone in range(self.zone_count)
        ]

    def shortest_path_zones(self, a: int, b: int) -> Iterable[int]:
        current = a
        yield current
        while current != b:
            candidates = [self.move(current, action) for action in self.valid_actions(current)]
            candidates = [candidate for candidate in candidates if candidate != current]
            if not candidates:
                return
            current = min(candidates, key=lambda candidate: self.hex_distance(candidate, b))
            yield current

    def hex_corners(self, zone: int, radius: float = 0.92) -> List[Tuple[float, float]]:
        x, y = self.coord(zone)
        return [
            (
                x + radius * math.cos(math.radians(30 + 60 * corner)),
                y + radius * math.sin(math.radians(30 + 60 * corner)),
            )
            for corner in range(6)
        ]

    def _cube(self, zone: int) -> Tuple[int, int, int]:
        row, col = self.to_rc(zone)
        cube_x = col - (row - (row & 1)) // 2
        cube_z = row
        cube_y = -cube_x - cube_z
        return cube_x, cube_y, cube_z

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols
