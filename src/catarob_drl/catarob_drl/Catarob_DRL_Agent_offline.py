import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
from catarob_drl_interfaces.msg import CatarobState
import sys
#sys.path.append('/home/ros2/dylan_ws/src/')     # uncomment for use on the catarob
import TD7
class CatarobDRLAgentNodeOffline(Node):
    def __init__(self, testing_mode=False):
        super().__init__('catarob_drl_agent_node')

        # Update parameters
        self.declare_parameter('state_dim', 10)  # Update based on new state representation
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('max_action', [2.0, 0.8])  # [linear velocity, angular velocity]
        self.declare_parameter('model_path', '/home/ros2/dylan_ws/src/model_10hz_R1')   # TODO add path

        # Declare parameters for model paths
        self.declare_parameter('actor_model_path', 'default_actor_path')
        self.declare_parameter('encoder_model_path', 'default_encoder_path')

        # Get model path parameters
        self.actor_model_path = self.get_parameter('actor_model_path').value
        self.encoder_model_path = self.get_parameter('encoder_model_path').value

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
            '/platform/cmd_vel',
            qos_profile
        )

        self.get_logger().info('Catarob DRL Agent Node initialized')

        # Timer for training loop (unchanged)
        #self.create_timer(0.25, self.training_loop)  # 4 Hz to match sensor data rate
        self.task_completed = False
        self.completion_time = None
        self.create_timer(0.1, self.check_completion)  # Timer to check completion status
        # Initialize state variables (unchanged)
        self.previous_state = None
        self.previous_action = None

    def load_model(self):
        try:
            # Load only the actor and encoder
            # self.agent.actor.load_state_dict(torch.load(self.model_path + "/model_iter_500000_actor.zip", map_location=torch.device('cpu'), weights_only=True ))
            # self.agent.fixed_encoder.load_state_dict(torch.load(self.model_path + "/model_iter_500000_encoder.zip", map_location=torch.device('cpu'), weights_only=True))
            # self.get_logger().info(f"Actor and encoder loaded successfully from {self.model_path}")
            # Load only the actor and encoder
            self.agent.actor.load_state_dict(torch.load(self.actor_model_path, map_location=torch.device('cpu')))
            self.agent.fixed_encoder.load_state_dict(torch.load(self.encoder_model_path, map_location=torch.device('cpu')))
            self.get_logger().info(f"Actor loaded from {self.actor_model_path}")
            self.get_logger().info(f"Encoder loaded from {self.encoder_model_path}")
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
        if msg.done:
            self.task_completed = True
            self.completion_time = self.get_clock().now()
            self.get_logger().info("Task completed! Publishing zero action for 1 second.")
            self.publish_action([0.0, 0.0])
        else:
            action = self.agent.select_action(state, use_checkpoint=self.testing_mode, use_exploration=False)
            self.publish_action(action)

        # # Select action using the TD7 agent
        # action = self.agent.select_action(state, use_checkpoint=self.testing_mode, use_exploration=False)
        # if msg.done:
        #     action = [0.0, 0.0]

        # # Publish action
        # self.publish_action(action)
        
    def publish_action(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.action_pub.publish(cmd_vel)
        self.get_logger().debug(f"Published action: linear={action[0]}, angular={action[1]}")

    def check_completion(self):
        if self.task_completed:
            current_time = self.get_clock().now()
            if (current_time - self.completion_time).nanoseconds / 1e9 >= 1.0:
                self.get_logger().info("Task finished. Shutting down the node.")
                self.destroy_node()
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    catarob_drl_agent_node = CatarobDRLAgentNodeOffline()
    rclpy.spin(catarob_drl_agent_node)
    catarob_drl_agent_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()