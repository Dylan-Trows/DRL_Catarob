import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import math
import numpy as np
from .Waypoint_manager import WaypointManager
from catarob_drl_interfaces.msg import VRXStepData

class RealCatarobController(Node):
    def __init__(self):
        super().__init__('real_catarob_controller')

        # Define QoS profiles
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
        
        self.waypoint_manager = WaypointManager()
        # Add your real waypoints here
        # TODO
        self.waypoint_manager.add_waypoint(-33.722499646, 150.674243934, 1.161502034)

        # Subscriptions
        self.create_subscription(NavSatFix, '/sensor/emlid_gps_fix', self.gps_callback, sensor_qos)
        self.create_subscription(Float64, '/sensors/mag_heading', self.heading_callback, sensor_qos)
        self.create_subscription(Twist, '/catarob/cmd_vel', self.action_callback, reliable_qos)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'catarob/cmd_vel', reliable_qos)
        self.step_publisher = self.create_publisher(VRXStepData, '/catarob/step_data', reliable_qos)

        # Control loop timer (4Hz to match sensor data rate)
        self.create_timer(0.25, self.control_loop)

        # Initialize variables
        self.current_heading = None
        self.gps_data = None
        self.current_waypoint = None
        self.last_action = None
        self.task_finished = False
        self.heading_error = 0.0
        self.previous_distance = None
        self.maximum_distance = None
        self.alpha = 1
        self.beta = 1
        self.max_speed = 2

    def gps_callback(self, msg):
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
            # Success!

    def heading_callback(self, msg):
        self.current_heading = msg.data

    def action_callback(self, msg):
        self.last_action = [msg.linear.x, msg.angular.z]

    def get_reward(self):
        reward = 0.0
        if self.current_waypoint is None or self.gps_data is None or self.last_action is None:
            return 0.0

        # Distance Reward
        l_arc = self.previous_distance * math.radians(abs(self.heading_error))
        normalized_d_target = self.previous_distance / self.maximum_distance
        normalized_l_arc = l_arc / (math.pi * self.previous_distance)
        w_arc, w_dist = 0.5, 0.5
        distance = w_arc * normalized_l_arc + w_dist * normalized_d_target
        distance_reward = -self.alpha * math.tanh(distance - 1)
        reward += distance_reward

        # Velocity reward
        forward_speed = self.last_action[0]
        speed_heading_reward = (forward_speed * math.cos(math.radians(self.heading_error))) / self.max_speed
        movement_reward = self.beta * speed_heading_reward
        reward += movement_reward

        # Energy usage penalty
        energy_usage = np.sum(np.abs(self.last_action))
        energy_penalty = -0.1 * energy_usage
        reward += energy_penalty

        if self.task_finished:
            reward += 100

        return reward

    def control_loop(self):
        if self.gps_data is None or self.current_heading is None:
            return

        self.current_waypoint = self.waypoint_manager.get_current_waypoint()

        if self.last_action is not None:                                # TODO agent to publish or controller ?
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = self.last_action[0]
            cmd_vel_msg.angular.z = self.last_action[1]
            self.cmd_vel_pub.publish(cmd_vel_msg)

        reward = self.get_reward()

        if self.gps_data is not None and self.current_waypoint is not None:
            step_data = RealCatarobStepData()
            step_data.gps_data = self.gps_data
            step_data.current_heading = self.current_heading
            step_data.current_waypoint = self.current_waypoint
            step_data.reward = reward
            step_data.task_finished = self.task_finished
            step_data.heading_error = self.heading_error

            self.step_publisher.publish(step_data)

    def reset_episode(self):                                            #TODO Implement logic for environment reset
        self.gps_data = None
        self.current_heading = None
        self.current_waypoint = None
        self.last_action = None
        self.task_finished = False
        self.waypoint_manager.reset()

def main(args=None):
    rclpy.init(args=args)
    controller = RealCatarobController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()