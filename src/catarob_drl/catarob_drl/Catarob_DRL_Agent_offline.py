import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
from catarob_drl_interfaces.msg import CatarobStepData
from .TD7 import Agent as TD7Agent
class CatarobDRLAgentNodeOffline(Node):
    def __init__(self, testing_mode=False):
        super().__init__('catarob_drl_agent_node')

        # Update parameters
        self.declare_parameter('state_dim', 10)  # Update based on new state representation
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('max_action', [2.0, 0.8])  # [linear velocity, angular velocity]
        self.declare_parameter('algorithm', 'TD3')
        self.declare_parameter('model_path', '/path/to/your/saved/model')   # TODO add path

        self.declare_parameter('stage', 'Test')
        self.testing_mode = testing_mode

        # Initialize TD7 agent
        self.agent = TD7Agent(self.state_dim, self.action_dim, self.max_action)
        self.load_model()

        # Get parameters
        self.state_dim = self.get_parameter('state_dim').value
        self.action_dim = self.get_parameter('action_dim').value
        self.max_action = self.get_parameter('max_action').value

        # QoS profile       
        qos_profile = QoSProfile(                                                                                                           #TODO change the QoS profile for the catarob
            reliability=ReliabilityPolicy.BEST_EFFORT,         
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Update subscriptions and publishers
        self.step_data_sub = self.create_subscription(
            CatarobStepData,
            '/catarob/step_data',
            self.step_data_callback,
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
    
    def step_data_callback(self, msg):
        # Update state representation
        gps_data = np.array(msg.gps_data, dtype=np.float32)
        current_heading = np.array([msg.current_heading], dtype=np.float32)
        current_waypoint = np.array(msg.current_waypoint, dtype=np.float32)
        heading_error = np.array([msg.heading_error], dtype=np.float32)
        
        current_state = np.concatenate([gps_data, current_heading, current_waypoint, heading_error])

        reward = msg.reward
        done = msg.task_finished

        state = np.concatenate(msg)
        # Update previous state
        self.previous_state = current_state

        # Select action using the TD7 agent
        action = self.agent.select_action(state, use_checkpoint=self.testing_mode, use_exploration=False)
        if done:
            action = [0.0, 0.0]
        self.publish_action(action)

        # Update previous action
        self.previous_action = action
            
        self.last_reward = msg.reward
        
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