import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

class DRLAgentNode(Node):
    def __init__(self):
        super().__init__('drl_agent_node')
        #TODO change between DDPG, TD3 AND TD3_BC
        self.agent = DDPG(state_dim, action_dim, max_action)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.logger = DataLogger()

        self.state_sub = self.create_subscription(
            Float32MultiArray, 'current_state', self.state_callback, 10
        )
        self.action_pub = self.create_publisher(
            Twist, 'cmd_vel', 10
        )
        self.training_timer = self.create_timer(
            0.1, self.training_loop
        )
    def state_callback(self, msg):
        state = np.array(msg.data)