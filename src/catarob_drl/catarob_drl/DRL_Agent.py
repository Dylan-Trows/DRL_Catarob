import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
from .Replay_Buffer import ReplayBuffer
from .TD3_test import TD3
from .DDPG_test import DDPG
from .TD3_BC_test import TD3_BC
from .Storage_manager import StorageManager
from .Training_Logger import DataLogger
from catarob_drl_interfaces.msg import VRXStepData
    
"""class PerformanceEvaluator:
    def __init__(self):
        self.total_reward = 0
        self.episode_count = 0
        self.step_count = 0

    def log_step(self, reward):
        self.total_reward += reward
        self.step_count += 1

    def log_episode(self):
        self.episode_count += 1

    def get_average_reward(self):
        return self.total_reward / self.step_count if self.step_count > 0 else 0

    def reset(self):
        self.total_reward = 0
        self.step_count = 0"""

class DRLAgentNode(Node):
    def __init__(self, testing_mode=False):
        super().__init__('drl_agent_node')

        # TODO make a hyperparameters.py program to handle the settings of the algorithms
        # Declare parameters
        self.declare_parameter('state_dim', 12)
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('max_action', 50.0)      #TODO determine the max thruster values of the CATAROB etc.
        self.declare_parameter('algorithm', 'TD3')
        self.declare_parameter('stage','Test')
        self.testing_mode = testing_mode

        # Get parameters
        self.state_dim = self.get_parameter('state_dim').value
        self.action_dim = self.get_parameter('action_dim').value
        self.max_action = self.get_parameter('max_action').value
        self.algorithm = self.get_parameter('algorithm').value

        # Initialize DRL algorithm
        if self.algorithm == 'TD3':
            self.agent = TD3(self.state_dim, self.action_dim, self.max_action)
        elif self.algorithm == 'TD3_BC':
            self.agent = TD3_BC(self.state_dim, self.action_dim, self.max_action)
        elif self.algorithm == 'DDPG':
            self.agent = DDPG(self.state_dim, self.action_dim, self.max_action)
        else:
            self.get_logger().error(f"Unknown algorithm: {self.algorithm}")
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.replay_buffer = ReplayBuffer(state_dim=self.state_dim, action_dim=self.action_dim)
        #self.performance_evaluator = PerformanceEvaluator()

        # Initialize StorageManager and DataLogger
        #base_path = self.get_parameter('base_path').value
        self.storage_manager = StorageManager("/home/dylan_trows/datalog_test", self.algorithm, self.get_parameter('stage').value)             #TODO add a "stage" for training/testing
        self.data_logger = DataLogger("/home/dylan_trows/datalog_test", self.algorithm, self.get_parameter('stage').value)

        # using QoS for a robust communication between nodes
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Add custom StepData custom message 
        self.step_data_sub = self.create_subscription(
            VRXStepData,
            '/vrx/step_data',
            self.step_data_callback,
            qos_profile
        )

        # Action Publishers for the left + right thrusters 
        self.action_pub = self.create_publisher(
            Float64MultiArray,
            '/vrx/action',
            qos_profile
        )

        # Timer for training loop
        self.create_timer(0.1, self.training_loop)

        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.total_timesteps = 0
        self.max_timesteps = 1000000
        self.episode_step = 0
        self.episode_count = 0
    
    def step_data_callback(self, msg):
        gps_data = np.array(msg.gps_data, dtype=np.float32)
        imu_data = np.array(msg.imu_data, dtype=np.float32)
        current_waypoint = np.array(msg.current_waypoint, dtype=np.float32)
        current_heading = np.array([msg.current_heading], dtype=np.float32)
        heading_error = np.array([msg.heading_error], dtype=np.float32)
        
        self.current_state = np.concatenate([gps_data, imu_data, current_waypoint, current_heading, heading_error])
        self.last_reward = msg.reward
        
        if self.last_action is not None:
            self.store_transition(self.last_action, self.last_reward, msg.task_finished)
        
        if msg.task_finished:
            self.end_episode()
    
    def store_transition(self, action, reward, done):
        self.replay_buffer.add(self.current_state, action, self.current_state, reward, done)         #  Stores the current transition (state, action, next_state, reward, done) in the replay buffer.
        #self.performance_evaluator.log_step(reward)                                                 #  Logs the step's reward for performance evaluation.
        self.episode_step += 1

    def end_episode(self):
        self.episode_count += 1
        self.get_logger().info(f"Episode {self.episode_count} finished. Steps: {self.episode_step}")
        self.episode_step = 0
        # Add other episode-end logic

    def training_loop(self):                                                    # called on timer 
        if self.current_state is None:                                          # error check (if sim hasnt started yet ?)
            return


        # # Exploration vs Exploitation
        # if self.total_timesteps < self.max_timesteps:                           #TODO set self.max_timesteps                           
        #     action = self.select_action(self.current_state, add_noise=False)     # encouraging exploration
        # else:
        #     action = self.select_action(self.current_state, add_noise=False)    # excouraging exploitation

        action = self.select_action(self.current_state, add_noise=False) 
        # Publish action
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        print("Action : ", action_msg.data)
        self.action_pub.publish(action_msg)

        # Update training state
        self.last_action = action
        self.total_timesteps += 1

        # Train the agent
        if self.testing_mode or (self.total_timesteps % 50 == 0 and self.replay_buffer.size > 256):    # trains periodically once enough samples in buffer
            print("Training now")
            actor_loss, critic_loss = self.train()                                                        #TODO set self.total_timesteps + replay buffer length (batch size)
            print("training done")
            self.data_logger.log_training_info(self.episode_step, actor_loss, critic_loss, self.total_timesteps / self.max_timesteps)

        # Save model periodically
        if self.episode_step % 1000 == 0:                                       #TODO get self.episode_step from VRX_Controller with state info
            self.storage_manager.save_model(self.agent, self.episode_step)
            self.storage_manager.save_replay_buffer(self.replay_buffer, self.episode_step)
            self.storage_manager.save_metadata({
                'total_timesteps': self.total_timesteps,
                'episode': self.episode_step
            }, self.episode_step)

    def select_action(self, state, add_noise=False):                             # gets action from current state
        action = self.agent.select_action(np.array(state))
        if add_noise:                                                           # adds gaussian noise to encourage exploration
            noise = np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action) # clipped to stay within allowed range
        return action

    def train(self, batch_size=256):                                            #TODO set batch_size
        actor_loss, critic_loss = self.agent.train(self.replay_buffer, batch_size)                        # training algorithm set in the DRL algorithm files
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
    drl_agent_node = DRLAgentNode()
    rclpy.spin(drl_agent_node)
    drl_agent_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()