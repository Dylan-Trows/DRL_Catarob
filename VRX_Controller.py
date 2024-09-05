import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from vrx_gazebo.msg import Task
import numpy as np
import torch

from Replay_Buffer import ReplayBuffer
from TD3_test import TD3
from DDPG_test import DDPG
from TD3_BC_test import TD3_BC

class VRXController(Node):
    def __init__(self):
        super().__init__('vrx_controller')

        # Define state and action dimensions
        self.state_dim = 7  # 3 for GPS, 4 for IMU orientation
        self.action_dim = 2  # Forward velocity and angular velocity
        self.max_action = 1.0  # Assuming for now that actions are normalized between -1 and 1

        # Choose the DRL algorithm to use
        self.algorithm = 'TD3'  # Options: 'TD3', 'TD3_BC', 'DDPG'

        # Initialize the chosen algorithm
        if self.algorithm == 'TD3':
            self.drl_agent = TD3(self.state_dim, self.action_dim, self.max_action)
        elif self.algorithm == 'TD3_BC':
            self.drl_agent = TD3_BC(self.state_dim, self.action_dim, self.max_action)
        elif self.algorithm == 'DDPG':
            self.drl_agent = DDPG(self.state_dim, self.action_dim, self.max_action)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim=self.state_dim, action_dim=self.action_dim)
        
        # Subscribe to ROS2 subscribers and publishers
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        self.create_subscription(Task, '/vrx/task/info', self.task_info_callback, 10)
        
        # Publisher for control commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/wamv/cmd_vel', 10)
        
        # Timer for control loop
        self.create_timer(0.1, self.control_loop)
        
        # Initialize state variables
        self.gps_data = None
        self.imu_data = None
        self.current_state = None
        self.episode_step = 0
        self.max_steps = 1000
        self.total_timesteps = 0
        self.max_timesteps = 1000000
        
    def gps_callback(self, msg):
        self.gps_data = [msg.latitude, msg.longitude, msg.altitude]
        
    def imu_callback(self, msg):
        self.imu_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        
    def task_info_callback(self, msg):
        self.task_state = msg.state
        
    def get_reward(self):
        # Implement your reward function here
        # This should be based on the specific objectives of your UASV task
        return 0.0
        
    def control_loop(self):
        if self.gps_data is None or self.imu_data is None:
            return
        
        # Combine sensor data to form the state
        self.current_state = np.array(self.gps_data + self.imu_data)
        
        # Training process
        if self.total_timesteps < self.max_timesteps:
            # Get action from the DRL agent
            if self.total_timesteps < 10000:  # Initial exploration phase
                action = np.random.uniform(-self.max_action, self.max_action, size=self.action_dim)
            else:
                action = self.drl_agent.select_action(self.current_state)
                if self.algorithm != 'TD3_BC':  # TD3_BC doesn't need exploration noise during training
                    noise = np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
                    action = np.clip(action + noise, -self.max_action, self.max_action)
            
            # Apply action to the robot
            cmd_vel = Twist()
            cmd_vel.linear.x = action[0]  # Forward velocity
            cmd_vel.angular.z = action[1]  # Angular velocity
            self.cmd_vel_pub.publish(cmd_vel)
            
            # Observe reward and next state
            reward = self.get_reward()
            done = (self.episode_step >= self.max_steps) or (self.task_state == Task.STATE_FINISHED)
            
            # Store transition in replay buffer
            self.replay_buffer.add(self.current_state, action, reward, self.current_state, done)
            
            # Train the DRL agent
            if self.total_timesteps % 50 == 0:
                self.drl_agent.train(self.replay_buffer, batch_size=256)
            
            self.episode_step += 1
            self.total_timesteps += 1
            
            if done:
                self.reset_simulation()
        else:
            # Use the trained policy without exploration
            action = self.drl_agent.select_action(self.current_state)
            cmd_vel = Twist()
            cmd_vel.linear.x = action[0]
            cmd_vel.angular.z = action[1]
            self.cmd_vel_pub.publish(cmd_vel)
        
    def reset_simulation(self):
        # Reset episode-specific variables
        self.episode_step = 0
        self.gps_data = None
        self.imu_data = None
        self.current_state = None
        
        # In VRX, you might need to wait for the next task to start
        # or implement a custom reset service

def main(args=None):
    rclpy.init(args=args)
    controller = VRXController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()