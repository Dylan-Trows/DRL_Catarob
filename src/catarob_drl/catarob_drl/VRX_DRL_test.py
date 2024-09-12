import unittest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import rclpy
from rclpy.node import Node
import os
import sys
import tempfile
import torch
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float32MultiArray, Float64MultiArray

# Add the current directory to the Python path
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your classes here
#from catarob_drl.DRL_Agent import DRLAgentNode
#from catarob_drl.VRX_Controller import VRXController
# from catarob_drl.Storage_manager import StorageManager
# from catarob_drl.Replay_Buffer import ReplayBuffer
# from catarob_drl.Training_Logger import DataLogger
# from catarob_drl.Waypoint_manager import WaypointManager
# from catarob_drl.TD3_test import TD3
# from catarob_drl_interfaces.msg import VRXStepData
#import VRXStepData

# Mock ROS2 specific modules
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['rclpy.qos'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['catarob_drl_interfaces.msg'] = MagicMock()

# Now import our modules
from catarob_drl.VRX_Controller import VRXController
from catarob_drl.DRL_Agent import DRLAgentNode

class TestDRLAgent(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     # Initialize rclpy once for all tests
    #     rclpy.init()

    # @classmethod
    # def tearDownClass(cls):
    #     # Shutdown rclpy after all tests
    #     rclpy.shutdown()

    @patch('catarob_drl.DRL_Agent.Node')
    @patch('catarob_drl.DRL_Agent.TD3')
    @patch('catarob_drl.DRL_Agent.ReplayBuffer')
    @patch('catarob_drl.DRL_Agent.StorageManager')
    @patch('catarob_drl.DRL_Agent.DataLogger')
    def setUp(self, mock_data_logger, mock_storage_manager, mock_replay_buffer, mock_td3, mock_node):
        self.agent = MagicMock(spec=DRLAgentNode)
        # self.agent.agent = mock_td3.return_value
        # self.agent.get_logger = MagicMock()
        # self.agent.create_subscription = MagicMock()
        # self.agent.create_publisher = MagicMock()
        # self.agent.create_timer = MagicMock()

        # Set up mock objects
        self.agent.agent = mock_td3.return_value
        self.agent.replay_buffer = mock_replay_buffer.return_value
        self.agent.storage_manager = mock_storage_manager.return_value
        self.agent.data_logger = mock_data_logger.return_value
        
        # Set up other attributes
        self.agent.current_state = None
        self.agent.last_action = None
        self.agent.last_reward = None
        self.agent.total_timesteps = 0
        self.agent.episode_step = 0
        self.agent.episode_count = 0

    def test_initialization(self, *args):
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.agent)
        self.assertIsNotNone(self.agent.replay_buffer)
        self.assertIsNotNone(self.agent.storage_manager)
        self.assertIsNotNone(self.agent.data_logger)
        # self.assertEqual(self.agent.state_dim, 11)
        # self.assertEqual(self.agent.action_dim, 2)
        # self.assertEqual(self.agent.max_action, 5000.0)

    def test_select_action_with_noise(self):
        state = np.random.rand(self.agent.state_dim)
        action = self.agent.select_action(state, add_noise=True)
        self.assertEqual(action.shape, (self.agent.action_dim,))
        self.assertTrue(np.all(action >= -self.agent.max_action))
        self.assertTrue(np.all(action <= self.agent.max_action))

    def test_select_action_without_noise(self):
        state = np.random.rand(self.agent.state_dim)
        action = self.agent.select_action(state, add_noise=False)
        self.assertEqual(action.shape, (self.agent.action_dim,))
        self.assertTrue(np.all(action >= -self.agent.max_action))
        self.assertTrue(np.all(action <= self.agent.max_action))

    @patch('DRL_Agent.TD3')
    def test_training_loop_exploration(self, mock_td3):
        mock_td3.return_value.select_action.return_value = np.zeros(self.agent.action_dim)
        mock_td3.return_value.train.return_value = (0.1, 0.2)  # mock actor_loss and critic_loss
        
        self.agent.current_state = np.random.rand(self.agent.state_dim)
        self.agent.last_action = np.zeros(self.agent.action_dim)
        self.agent.last_reward = 0.0
        self.agent.total_timesteps = 1000 # Less than max_timesteps
        self.agent.replay_buffer.add(np.zeros(self.agent.state_dim), np.zeros(self.agent.action_dim), 
                                    np.zeros(self.agent.state_dim), 0.0, False)

        self.agent.training_loop()
        
        self.assertTrue(mock_td3.return_value.select_action.called)
        self.assertFalse(mock_td3.return_value.train.called)  # Should not train yet

    @patch('catarob_drl.DRL_Agent.TD3')
    def test_training_loop_exploitation(self, mock_td3):
        mock_td3.return_value.select_action.return_value = np.zeros(self.agent.action_dim)
        mock_td3.return_value.train.return_value = (0.1, 0.2)  # mock actor_loss and critic_loss
        
        self.agent.current_state = np.random.rand(self.agent.state_dim)
        self.agent.last_action = np.zeros(self.agent.action_dim)
        self.agent.last_reward = 0.0
        self.agent.total_timesteps = 1000  # Equal to max_timesteps
        self.agent.replay_buffer.add(np.zeros(self.agent.state_dim), np.zeros(self.agent.action_dim), 
                                    np.zeros(self.agent.state_dim), 0.0, False)

        self.agent.training_loop()
        
        self.assertTrue(mock_td3.return_value.select_action.called)
        self.assertTrue(mock_td3.return_value.train.called)

    def test_step_data_callback(self):
        mock_msg = MagicMock()
        mock_msg.gps_data = [0.0, 0.0, 0.0]
        mock_msg.imu_data = [0.0, 0.0, 0.0, 1.0]
        mock_msg.current_waypoint = [1.0, 1.0, 0.0, 0.0]
        mock_msg.current_heading = 0.0
        mock_msg.heading_error = 0.0
        mock_msg.reward = 1.0
        mock_msg.task_finished = False

        self.agent.last_action = np.zeros(self.agent.action_dim)
        self.agent.step_data_callback(mock_msg)

        self.assertIsNotNone(self.agent.current_state)
        self.assertEqual(self.agent.last_reward, 1.0)
        self.assertEqual(len(self.agent.replay_buffer), 1)

    def test_end_episode(self):
        initial_episode_count = self.agent.episode_count
        self.agent.end_episode()
        self.assertEqual(self.agent.episode_count, initial_episode_count + 1)
        self.assertEqual(self.agent.episode_step, 0)

class TestVRXController(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     # Initialize rclpy once for all tests
    #     rclpy.init()

    # @classmethod
    # def tearDownClass(cls):
    #     # Shutdown rclpy after all tests
    #     rclpy.shutdown()


    @patch('catarob_drl.VRX_Controller.Node')
    def setUp(self, mock_node):
        self.controller = VRXController()

    def test_initialization(self):
        self.assertIsNotNone(self.controller.waypoint_manager)
        self.assertEqual(self.agent.current_heading, 0.0)
        self.assertIsNone(self.agent.gps_data)
        self.assertIsNone(self.node.imu_data)
        self.assertIsNone(self.node.current_waypoint)
        self.assertIsNone(self.node.last_action)
        self.assertFalse(self.node.task_finished)
        self.assertEqual(self.node.heading_error, 0.0)

    def test_gps_callback(self):
        mock_msg = MagicMock()
        mock_msg.latitude = 1.0
        mock_msg.longitude = 2.0
        mock_msg.altitude = 3.0
        self.node.gps_callback(mock_msg)
        self.assertEqual(self.node.gps_data, [1.0, 2.0, 3.0])
        self.node.imu_callback(mock_msg)
        self.assertIsNotNone(self.node.imu_data)
        self.assertIsNotNone(self.node.current_heading)

    def test_imu_callback(self):
        mock_msg = MagicMock()
        mock_msg.orientation.x = 5.0
        mock_msg.orientation.y = 6.0
        mock_msg.orientation.z = 7.0
        mock_msg.orientation.w = 1.0
        self.node.imu_callback(mock_msg)
        self.assertEqual(self.node.imu_data, [5.0, 6.0, 7.0, 1.0])
        self.assertEqual(self.node.current_heading, 138.9575531141279)

    def test_action_callback(self):
        mock_msg = MagicMock()
        mock_msg.data = [100.0, -100.0]
        self.node.action_callback(mock_msg)
        self.assertEqual(self.node.last_action, [100.0, -100.0])

    def test_get_reward(self):
        self.node.heading_error = 0.0
        self.node.task_finished = False
        reward = self.node.get_reward()
        self.assertEqual(reward, 1.0)

        self.node.heading_error = 180.0
        reward = self.node.get_reward()
        self.assertAlmostEqual(reward, -1.0, places=5)

        self.node.task_finished = True
        reward = self.node.get_reward()
        self.assertAlmostEqual(reward, 99.0, places=5)

    @patch('catarob_drl.VRX_Controller.Float64')
    @patch('catarob_drl.VRX_Controller.VRXStepData')
    def test_control_loop(self, mock_vrx_step_data, mock_float64):
        self.node.gps_data = [0.0, 0.0, 0.0]
        self.node.imu_data = [0.0, 0.0, 0.0, 1.0]
        self.node.last_action = [100.0, -100.0]
        self.node.current_waypoint = [1.0, 1.0, 0.0, 0.0]
        self.node.control_loop()
        self.assertTrue(mock_float64.called)
        self.assertTrue(mock_vrx_step_data.called)

# class TestStorageManager(unittest.TestCase):
#     def setUp(self):
#         self.temp_dir = tempfile.mkdtemp()
#         self.storage_manager = StorageManager(self.temp_dir, "TD3", "1")

#     def tearDown(self):
#         for root, dirs, files in os.walk(self.temp_dir, topdown=False):
#             for name in files:
#                 os.remove(os.path.join(root, name))
#             for name in dirs:
#                 os.rmdir(os.path.join(root, name))
#         os.rmdir(self.temp_dir)

#     def test_save_and_load_model(self):
#         model = TD3(state_dim=4, action_dim=2, max_action=1.0)
#         self.storage_manager.save_model(model, 100)
        
#         loaded_model = TD3(state_dim=4, action_dim=2, max_action=1.0)
#         self.storage_manager.load_model(loaded_model, 100)
        
#         # Check if all parameters are equal
#         for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
#             self.assertTrue(torch.equal(p1, p2))

#     def test_save_and_load_replay_buffer(self):
#         replay_buffer = ReplayBuffer(state_dim=4, action_dim=2, max_size=1000)
#         replay_buffer.add(np.zeros(4), np.zeros(2), np.ones(4), 1.0, False)
        
#         self.storage_manager.save_replay_buffer(replay_buffer, 100)
#         loaded_buffer = self.storage_manager.load_replay_buffer(100)
        
#         self.assertEqual(replay_buffer.size, loaded_buffer.size)
#         np.testing.assert_array_equal(replay_buffer.state, loaded_buffer.state)

#     def test_save_and_load_metadata(self):
#         metadata = {'total_timesteps': 1000, 'episode': 50}
#         self.storage_manager.save_metadata(metadata, 100)
#         loaded_metadata = self.storage_manager.load_metadata(100)
#         self.assertEqual(metadata, loaded_metadata)

# class TestReplayBuffer(unittest.TestCase):
#     def setUp(self):
#         self.replay_buffer = ReplayBuffer(state_dim=4, action_dim=2, max_size=1000)

#     def test_add_and_sample(self):
#         state = np.random.rand(4)
#         action = np.random.rand(2)
#         next_state = np.random.rand(4)
#         reward = np.random.rand()
#         done = False

#         self.replay_buffer.add(state, action, next_state, reward, done)
        
#         batch = self.replay_buffer.sample(1)
#         self.assertEqual(len(batch), 5)  # state, action, next_state, reward, not_done
#         self.assertEqual(batch[0].shape, (1, 4))  # state
#         self.assertEqual(batch[1].shape, (1, 2))  # action

#     def test_buffer_overflow(self):
#         for _ in range(1500):  # More than max_size
#             self.replay_buffer.add(np.zeros(4), np.zeros(2), np.zeros(4), 0, False)
        
#         self.assertEqual(self.replay_buffer.size, 1000)

# class TestDataLogger(unittest.TestCase):
#     def setUp(self):
#         self.temp_dir = tempfile.mkdtemp()
#         self.data_logger = DataLogger(self.temp_dir, "TD3", "1")

#     def tearDown(self):
#         for root, dirs, files in os.walk(self.temp_dir, topdown=False):
#             for name in files:
#                 os.remove(os.path.join(root, name))
#             for name in dirs:
#                 os.rmdir(os.path.join(root, name))
#         os.rmdir(self.temp_dir)

#     def test_log_episode(self):
#         self.data_logger.log_episode(1, 100.0, 500, True)
#         self.assertEqual(len(self.data_logger.episode_rewards), 1)
#         self.assertEqual(len(self.data_logger.episode_lengths), 1)
#         self.assertEqual(len(self.data_logger.episode_successes), 1)

#     def test_log_training_info(self):
#         self.data_logger.log_training_info(1, 0.1, 0.2, 0.9)
#         training_log_file = os.path.join(self.data_logger.log_dir, f"{self.data_logger.algorithm}_training_info_stage{self.data_logger.stage}_{self.data_logger.current_time}.csv")
#         self.assertTrue(os.path.exists(training_log_file))

#     def test_log_test_results(self):
#         self.data_logger.log_test_results(100, 0.8, 150.0, 450)
#         test_log_file = os.path.join(self.data_logger.log_dir, f"{self.data_logger.algorithm}_test_results_stage{self.data_logger.stage}_{self.data_logger.current_time}.txt")
#         self.assertTrue(os.path.exists(test_log_file))

#     def test_get_current_performance(self):
#         for i in range(150):
#             self.data_logger.log_episode(i, i, i, i % 2 == 0)
#         avg_reward, success_rate = self.data_logger.get_current_performance()
#         self.assertAlmostEqual(avg_reward, 99.5, places=1)
#         self.assertAlmostEqual(success_rate, 0.5, places=1)

# class TestWaypointManager(unittest.TestCase):
#     def setUp(self):
#         self.waypoint_manager = WaypointManager()

#     def test_add_and_get_waypoint(self):
#         self.waypoint_manager.add_waypoint(2.0, 2.0, 1.0)
#         waypoint = self.waypoint_manager.get_current_waypoint()
#         self.assertEqual(waypoint, (2.0, 2.0, 1.0, None))
    
#     def test_update_position(self):
#         self.waypoint_manager.add_waypoint(5.0, 5.0, 1.0)
#         self.waypoint_manager.update_position(5.00000001, 5.0000001, 1.0)
#         self.assertFalse(self.waypoint_manager.has_more_waypoints())

#     def test_calculate_distance(self):
#         distance = self.waypoint_manager.calculate_distance(0.0, 0.0, 1.0, 1.0)
#         self.assertAlmostEqual(distance, 157249.38, places=2)

#     def test_calculate_bearing(self):
#         bearing = self.waypoint_manager.calculate_bearing(0.0, 0.0, 1.0, 1.0)
#         self.assertAlmostEqual(bearing, 44.99563, places=5)

    if __name__ == '__main__':
        unittest.main()
