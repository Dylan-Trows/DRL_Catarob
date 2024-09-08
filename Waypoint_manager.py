#implement WayPoint manager class to manage GPS waipoint tracking for the UASV 

import math
from typing import List, Tuple

class Waypoint:
    def __init__(self, latitude: float, longitude: float, altitude: float, heading: float = None):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.heading = heading

class WaypointManager:
    def __init__(self):
        self.waypoints: List[Waypoint] = []
        self.current_index: int = 0
        self.reached_threshold: float = 5.0  # meters

    def add_waypoint(self, latitude: float, longitude: float, altitude: float, heading: float = None):
        """Add a new waypoint to the list."""
        self.waypoints.append(Waypoint(latitude, longitude, altitude, heading))

    def get_current_waypoint(self) -> Tuple[float, float, float, float]:
        """Get the current waypoint data."""
        if self.current_index < len(self.waypoints):
            wp = self.waypoints[self.current_index]
            return wp.latitude, wp.longitude, wp.altitude, wp.heading
        return None
    
    def update_position(self, current_lat: float, current_lon: float, current_alt: float):
        """Update the current position and check if the waypoint is reached."""
        if self.current_index < len(self.waypoints):
            wp = self.waypoints[self.current_index]
            distance = self.calculate_distance(current_lat, current_lon, wp.latitude, wp.longitude)
            if distance < self.reached_threshold:
                self.move_to_next_waypoint()
    
    def move_to_next_waypoint(self):
        """Move to the next waypoint."""
        self.current_index += 1
    
    def reset(self):
        """Reset the waypoint manager."""
        self.current_index = 0

    def has_more_waypoints(self) -> bool:
        """Check if there are more waypoints."""
        return self.current_index < len(self.waypoints)
    
    def get_desired_heading(self, current_lat: float, current_lon: float) -> float:
        """Calculate the desired heading to the current waypoint."""
        if self.current_index < len(self.waypoints):
            wp = self.waypoints[self.current_index]
            return self.calculate_bearing(current_lat, current_lon, wp.latitude, wp.longitude)
        return None
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the distance between two GPS coordinates using the Haversine formula."""
        R = 6371000  # Earth radius in meters (global average-value)

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c
    
    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the bearing between two GPS coordinates."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)

        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - \
            math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        theta = math.atan2(y, x)

        return (math.degrees(theta) + 360) % 360
