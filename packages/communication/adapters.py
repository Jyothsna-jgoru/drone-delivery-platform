from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class WaypointCommand:
    latitude: float
    longitude: float
    altitude: float
    speed_mps: float


class FlightCommandAdapter(ABC):
    """Interface boundary only; core simulation never sends hardware commands."""

    @abstractmethod
    def publish_waypoint(self, vehicle_id: str, command: WaypointCommand) -> None:
        raise NotImplementedError


class Ros2Adapter(FlightCommandAdapter):
    def publish_waypoint(self, vehicle_id: str, command: WaypointCommand) -> None:
        try:
            import rclpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ROS 2 adapter requested but rclpy is not installed") from exc
        raise RuntimeError("ROS 2 bridge must be explicitly configured before flight commands are enabled")


class Px4MavlinkAdapter(FlightCommandAdapter):
    def publish_waypoint(self, vehicle_id: str, command: WaypointCommand) -> None:
        try:
            import pymavlink  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("PX4 adapter requested but pymavlink is not installed") from exc
        raise RuntimeError("PX4 SITL bridge must be explicitly configured before commands are enabled")
