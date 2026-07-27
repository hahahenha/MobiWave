from __future__ import annotations

from typing import Any
import math


def nearest_assignments(
    snapshot: dict[str, Any],
    max_pickup_time: int | None = None,
) -> dict[int, int]:
    """Assign requests before MobiWave rebalances the remaining idle fleet."""

    network = snapshot["network"]
    time = int(snapshot["time"])
    if max_pickup_time is None:
        max_pickup_time = int(snapshot["config"].max_wait)
    vehicles = snapshot["vehicles"]
    requests = snapshot["requests"]
    idle_ids = [
        vehicle_id
        for vehicle_id, vehicle in vehicles.items()
        if vehicle.status == "idle"
    ]
    request_ids = sorted(
        requests,
        key=lambda request_id: (
            time - requests[request_id].created_at,
            requests[request_id].fare,
        ),
        reverse=True,
    )
    assignments: dict[int, int] = {}
    used_vehicles: set[int] = set()
    for request_id in request_ids:
        request = requests[request_id]
        best_vehicle = None
        best_cost = math.inf
        for vehicle_id in idle_ids:
            if vehicle_id in used_vehicles:
                continue
            vehicle = vehicles[vehicle_id]
            pickup = network.travel_time(vehicle.zone, request.origin, time)
            if pickup > max_pickup_time:
                continue
            cost = pickup - 0.08 * request.fare
            if cost < best_cost:
                best_cost = cost
                best_vehicle = vehicle_id
        if best_vehicle is not None:
            assignments[best_vehicle] = request_id
            used_vehicles.add(best_vehicle)
    return assignments
