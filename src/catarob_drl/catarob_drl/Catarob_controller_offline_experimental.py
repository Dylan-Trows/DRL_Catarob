import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from catarob_drl_interfaces.msg import CatarobState
from .Waypoint_manager import WaypointManager
import numpy as np
import math
from threading import Lock

class CatarobController(Node):
    def __init__(self):
        super().__init__('catarob_controller')

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Initialize WaypointManager
        self.waypoint_manager = WaypointManager()
        self.waypoint_manager.add_waypoint(-33.722499646, 150.674243934, 1.161502034)  # Example waypoint

        # Subscriptions
        self.create_subscription(NavSatFix, '/sensors/emlid_gps_fix', self.gps_callback, sensor_qos)
        self.create_subscription(Float64, '/sensors/mag_heading', self.heading_callback, sensor_qos)

        # Publishers
        self.state_publisher = self.create_publisher(CatarobState, '/catarob/state', reliable_qos)

        # Initialize variables
        self.current_heading = None
        self.gps_data = None
        self.current_waypoint = None
        self.last_action = None
        self.task_finished = False
        self.heading_error = 0.0
        self.previous_distance = None
        self.maximum_distance = None
        self.prev_x = None
        self.prev_y = None
        self.ref_lat = None
        self.ref_lon = None

        # Flags for tracking received sensor data
        self.gps_received = False
        self.heading_received = False

        # Lock for thread safety
        self.state_lock = Lock()

        # Timer for checking and publishing state
        self.create_timer(0.1, self.check_and_publish_state)  # 10 Hz

        self.get_logger().info('Catarob Controller Node initialized')

    def gps_callback(self, msg):
        with self.state_lock:
            self.gps_data = [msg.latitude, msg.longitude, msg.altitude]
            if self.ref_lat is None:
                self.ref_lat, self.ref_lon = msg.latitude, msg.longitude
            self.waypoint_manager.update_position(msg.latitude, msg.longitude, msg.altitude)
            self.gps_received = True

    def heading_callback(self, msg):
        with self.state_lock:
            self.current_heading = self.waypoint_manager.magnetic_to_true_heading(msg.data)
            self.heading_received = True

    def check_and_publish_state(self):
        with self.state_lock:
            if self.gps_received and self.heading_received:
                state = self.calculate_state()
                if state:
                    self.publish_state(state)
                    # Reset flags
                    self.gps_received = False
                    self.heading_received = False

    def calculate_state(self):
        if self.gps_data is None or self.current_heading is None:
            return None

        current_waypoint = self.waypoint_manager.get_current_waypoint()
        if current_waypoint is None:
            return None

        lat, lon, _ = self.gps_data
        waypoint_lat, waypoint_lon, _ = current_waypoint

        x, y = self.waypoint_manager.latlon_to_xy(lat, lon, self.ref_lat, self.ref_lon)
        waypoint_x, waypoint_y = self.waypoint_manager.latlon_to_xy(waypoint_lat, waypoint_lon, self.ref_lat, self.ref_lon)

        desired_heading = self.waypoint_manager.calculate_bearing(lat, lon, waypoint_lat, waypoint_lon)
        heading_error = self.waypoint_manager.calculate_heading_error(self.current_heading, desired_heading)

        distance_to_waypoint = self.waypoint_manager.calculate_distance(lat, lon, waypoint_lat, waypoint_lon)
        arc_length = distance_to_waypoint * math.radians(abs(heading_error))

        velocity = 0.0
        if self.prev_x is not None and self.prev_y is not None:
            velocity = self.waypoint_manager.calculate_velocity(self.prev_x, self.prev_y, x, y, 0.1)  # 0.1 seconds for 10 Hz

        done = distance_to_waypoint < 1.0 or not self.waypoint_manager.has_more_waypoints()

        self.prev_x, self.prev_y = x, y

        return {
            'x': round(x, 2),
            'y': round(y, 2),
            'heading': round(self.current_heading, 1),
            'heading_error': round(heading_error, 1),
            'distance_to_waypoint': round(distance_to_waypoint, 2),
            'arc_length': round(arc_length, 2),
            'velocity': round(velocity, 2),
            'waypoint_x': round(waypoint_x, 2),
            'waypoint_y': round(waypoint_y, 2),
            'done': done
        }

    def publish_state(self, state):
        msg = CatarobState()
        msg.x = state['x']
        msg.y = state['y']
        msg.heading = state['heading']
        msg.heading_error = state['heading_error']
        msg.distance_to_waypoint = state['distance_to_waypoint']
        msg.arc_length = state['arc_length']
        msg.velocity = state['velocity']
        msg.waypoint_x = state['waypoint_x']
        msg.waypoint_y = state['waypoint_y']
        msg.done = state['done']
        self.state_publisher.publish(msg)
        self.get_logger().debug(f"Published state: {state}")

    def reset_episode(self):
        with self.state_lock:
            self.gps_data = None
            self.current_heading = None
            self.ref_lat = None
            self.ref_lon = None
            self.prev_x = None
            self.prev_y = None
            self.gps_received = False
            self.heading_received = False
            self.waypoint_manager.reset()
        self.get_logger().info("Episode reset")

def main(args=None):
    rclpy.init(args=args)
    controller = CatarobController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()