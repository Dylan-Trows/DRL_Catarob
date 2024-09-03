import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from vrx_gazebo.msg import Task
import numpy as np
import torch

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
        
    