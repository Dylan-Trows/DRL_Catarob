import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import math
import numpy as np
from .Waypoint_manager import WaypointManager
from catarob_drl_interfaces.msg import CatarobState

class CatarobController(Node):
    def __init__(self):
        super().__init__('catarob_controller')

        # Define QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            #durability=DurabilityPolicy.VOLATILE
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.waypoint_manager = WaypointManager()
        # Add your real waypoints here
        # TODO
        self.waypoint_manager.add_waypoint(-33.722499646, 150.674243934, 1.161502034)

        # Subscriptions
        self.create_subscription(NavSatFix, '/sensors/emlid_gps_fix', self.gps_callback, sensor_qos)
        self.create_subscription(Float64, '/sensors/mag_heading', self.heading_callback, sensor_qos)
        self.create_subscription(Twist, '/catarob/cmd_vel', self.action_callback, sensor_qos)

        # Publishers
        self.state_publisher = self.create_publisher(CatarobState, '/catarob/state', reliable_qos)

        # Control loop timer (10Hz to match sensor data rate)
        self.create_timer(0.1, self.control_loop)

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

        self.get_logger().info('Catarob Controller Node initialized')

    """def gps_callback(self, msg):
        self.gps_data = [msg.latitude, msg.longitude, msg.altitude]
        self.waypoint_manager.update_position(msg.latitude, msg.longitude, msg.altitude)
        current_waypoint = self.waypoint_manager.get_current_waypoint()
        if current_waypoint:
            if self.maximum_distance is None and self.previous_distance is None:
                self.maximum_distance = self.waypoint_manager.calculate_distance(msg.latitude, msg.longitude, current_waypoint[0], current_waypoint[1])
                self.previous_distance = self.maximum_distance
            elif self.maximum_distance is not None:
                self.previous_distance = self.waypoint_manager.calculate_distance(msg.latitude, msg.longitude, current_waypoint[0], current_waypoint[1])

            desired_heading = self.waypoint_manager.get_desired_heading(msg.latitude, msg.longitude)
            if self.current_heading is not None:
                self.heading_error = self.waypoint_manager.calculate_heading_error(self.current_heading, desired_heading)                                       # Get Heading error for Reward function
        # Check if we've reached all waypoints
        if not self.waypoint_manager.has_more_waypoints():
            self.task_finished = True
            # Success!"""

    def gps_callback(self, msg):
        self.gps_data = [msg.latitude, msg.longitude, msg.altitude]
        if self.ref_lat is None:
            self.ref_lat, self.ref_lon = msg.latitude, msg.longitude
        self.waypoint_manager.update_position(msg.latitude, msg.longitude, msg.altitude)

    def heading_callback(self, msg):
        self.current_heading = self.waypoint_manager.magnetic_to_true_heading(msg.data)

    def action_callback(self, msg):
        self.last_action = [msg.linear.x, msg.angular.z]

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
    
    def reset_episode(self):
        self.gps_data = None
        self.current_heading = None
        self.current_waypoint = None
        self.last_action = None
        self.task_finished = False
        self.waypoint_manager.reset()
        self.prev_x = None
        self.prev_y = None
        self.ref_lat = None
        self.ref_lon = None
        self.get_logger().info("Episode reset")

    def control_loop(self):
        if self.gps_data is None or self.current_heading is None:
            return

        state = self.calculate_state()
        if state:
            self.publish_state(state)

def main(args=None):
    rclpy.init(args=args)
    controller = CatarobController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()