import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
from .Replay_Buffer import ReplayBuffer
from .TD3_test import TD3
from .DDPG_test import DDPG
from .TD3_BC_test import TD3_BC
from .Storage_manager import StorageManager
from .Training_Logger import DataLogger
from catarob_drl_interfaces.msg import CatarobStepData
from .TD7 import Agent as TD7Agent

class CatarobDRLAgentNode(Node):
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
        self.algorithm = self.get_parameter('algorithm').value

        # Initialize DRL algorithm (unchanged)
        if self.algorithm == 'TD3':
            self.agent = TD3(self.state_dim, self.action_dim, 1)
        elif self.algorithm == 'TD3_BC':
            self.agent = TD3_BC(self.state_dim, self.action_dim, 1)
        elif self.algorithm == 'DDPG':
            self.agent = DDPG(self.state_dim, self.action_dim, 1)
        else:
            self.get_logger().error(f"Unknown algorithm: {self.algorithm}")
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.replay_buffer = ReplayBuffer(state_dim=self.state_dim, action_dim=self.action_dim)

        # Initialize StorageManager and DataLogger (unchanged)
        self.storage_manager = StorageManager("/home/dylan_trows/datalog_test", self.algorithm, self.get_parameter('stage').value)          #TODO add a "stage" for training/testing
        self.data_logger = DataLogger("/home/dylan_trows/datalog_test", self.algorithm, self.get_parameter('stage').value)

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
        self.last_reward = None
        self.total_timesteps = 0
        self.max_timesteps = 1000000
        self.episode_step = 0
        self.episode_count = 0
        self.episode_reward = 0.0
    
    def step_data_callback(self, msg):
        # Update state representation
        gps_data = np.array(msg.gps_data, dtype=np.float32)
        current_heading = np.array([msg.current_heading], dtype=np.float32)
        current_waypoint = np.array(msg.current_waypoint, dtype=np.float32)
        heading_error = np.array([msg.heading_error], dtype=np.float32)
        
        current_state = np.concatenate([gps_data, current_heading, current_waypoint, heading_error])

        reward = msg.reward
        done = msg.task_finished

        # Store transition if we have a previous state and action
        if self.previous_state is not None and self.previous_action is not None:
            self.store_transition(self.previous_state, self.previous_action, reward, current_state, done)

        # Update previous state
        self.previous_state = current_state

        # Select and publish new action
        action = self.select_action(current_state)
        self.publish_action(action)

        # Update previous action
        self.previous_action = action

        # Update episode information
        self.episode_step += 1
        self.episode_reward += reward
            
        self.last_reward = msg.reward
        
        if done:
            self.end_episode()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, next_state, reward, done)         #  Stores the current transition (state, action, next_state, reward, done) in the replay buffer.
                                                                                                     #  Logs the step's reward for performance evaluation.

    def select_action(self, state, add_noise=False):                                                        # gets action from current state
        action = self.agent.select_action(np.array(state))
        if self.testing_mode:
            return action
        if add_noise:                                                                                       # adds gaussian noise to encourage exploration
            noise = np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)                             # clipped to stay within allowed range
        return action
    
    # The rest of the methods (store_transition, end_episode, training_loop, select_action, train, load_checkpoint)
    # remain largely unchanged. You may need to adjust the action publishing in the training_loop:

    def end_episode(self):
        self.episode_count += 1
        self.get_logger().info(f"Episode {self.episode_count} finished. Steps: {self.episode_step}, Reward: {self.episode_reward}")
        self.episode_step = 0
        self.episode_reward = 0
        self.previous_state = None
        self.previous_action = None

    def publish_action(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action.tolist()[0] * self.max_action
        cmd_vel.angular.z = action.tolist()[1] * 0.8
        self.action_pub(cmd_vel)

    def training_loop(self):                                                                         # called on timer 
        if self.current_state is None:                                                               # error check (if sim hasnt started yet ?)
            return
        
        self.total_timesteps += 1

        # Train the agent (unchanged)
        if not self.testing_mode and (self.total_timesteps % 50 == 0 and self.replay_buffer.size > 256):
            actor_loss, critic_loss = self.train()
            self.data_logger.log_training_info(self.episode_step, actor_loss, critic_loss, self.total_timesteps / self.max_timesteps)

        # Save model periodically (unchanged)
        if self.total_timesteps % 1000 == 0:
            self.storage_manager.save_model(self.agent, self.total_timesteps)
            self.storage_manager.save_replay_buffer(self.replay_buffer, self.total_timesteps)
            self.storage_manager.save_metadata({
                'total_timesteps': self.total_timesteps,
                'episode': self.episode_count
            }, self.total_timesteps)
        
    
    
    def train(self, batch_size=256):                                                                        #TODO set batch_size
        actor_loss, critic_loss = self.agent.train(self.replay_buffer, batch_size)                          # training algorithm set in the DRL algorithm files
        return actor_loss, critic_loss
    
    def load_checkpoint(self, episode):
        self.storage_manager.load_model(self.agent, episode)
        self.replay_buffer = self.storage_manager.load_replay_buffer(episode)
        metadata = self.storage_manager.load_metadata(episode)
        if metadata:
            self.total_timesteps = metadata['total_timesteps']
            self.episode = metadata['episode']

    

def main(args=None):
    rclpy.init(args=args)
    catarob_drl_agent_node = CatarobDRLAgentNode()
    rclpy.spin(catarob_drl_agent_node)
    catarob_drl_agent_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()