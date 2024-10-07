import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
from catarob_drl_interfaces.msg import CatarobState
import TD7
class CatarobDRLAgentNodeOffline(Node):
    def __init__(self, testing_mode=False):
        super().__init__('catarob_drl_agent_node')

        # Update parameters
        self.declare_parameter('state_dim', 10)  # Update based on new state representation
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('max_action', [2.0, 0.8])  # [linear velocity, angular velocity]
        self.declare_parameter('model_path', '/path/to/your/saved/model')   # TODO add path

        # Get parameters
        self.state_dim = self.get_parameter('state_dim').value
        self.action_dim = self.get_parameter('action_dim').value
        self.max_action = np.array(self.get_parameter('max_action').value)
        self.model_path = self.get_parameter('model_path').value
        self.testing_mode = testing_mode
        self.declare_parameter('stage', 'Test')
        self.testing_mode = testing_mode

        # Initialize TD7 agent
        self.agent = TD7.Agent(self.state_dim, self.action_dim, self.max_action)
        self.load_model()

        # QoS profile       
        qos_profile = QoSProfile(                                                                                                           #TODO change the QoS profile for the catarob
            reliability=ReliabilityPolicy.BEST_EFFORT,         
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions and publishers
        self.state_sub = self.create_subscription(
            CatarobState,
            '/catarob/state',
            self.state_callback,
            qos_profile
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/catarob/cmd_vel',
            qos_profile
        )

        self.get_logger().info('Catarob DRL Agent Node initialized')

        # Timer for training loop (unchanged)
        self.create_timer(0.25, self.training_loop)  # 4 Hz to match sensor data rate

        # Initialize state variables (unchanged)
        self.previous_state = None
        self.previous_action = None

    def load_model(self):
        try:
            self.agent.load(self.model_path)
            self.get_logger().info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")

    def state_callback(self, msg):
        # Process the state data
        state = np.array([
            msg.x,
            msg.y,
            msg.heading,
            msg.heading_error,
            msg.distance_to_waypoint,
            msg.arc_length,
            msg.velocity,
            msg.waypoint_x,
            msg.waypoint_y,
            int(msg.done)
        ])

        # Select action using the TD7 agent
        action = self.agent.select_action(state, use_checkpoint=self.testing_mode, use_exploration=False)
        if msg.done:
            action = [0.0, 0.0]

        # Publish action
        self.publish_action(action)
        
    def publish_action(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.action_pub.publish(cmd_vel)
        self.get_logger().debug(f"Published action: linear={action[0]}, angular={action[1]}")

def main(args=None):
    rclpy.init(args=args)
    catarob_drl_agent_node = CatarobDRLAgentNodeOffline()
    rclpy.spin(catarob_drl_agent_node)
    catarob_drl_agent_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()