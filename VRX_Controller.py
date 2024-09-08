# # VRXStepData.msg
# float32[] state
# float32 reward
# bool done
# int32 episode_step

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float32MultiArray, Float64MultiArray
from vrx_gazebo.msg import Task                         #TODO Implement Task ROS2 Topic?
import numpy as np
from Waypoint_manager import WaypointManager            #TODO Implement Waypoint manager class
import VRXStepData 

class VRXController(Node):
    def __init__(self):
        super().__init__('vrx_controller')

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

        # Use sensor_qos for sensor data subscriptions
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, sensor_qos)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, sensor_qos)
        
        # Use reliable_qos for task info and action subscriptions
        self.create_subscription(Task, '/vrx/task/info', self.task_info_callback, reliable_qos)                       #TODO
        self.create_subscription(Float64MultiArray, '/vrx/action', self.action_callback, reliable_qos)                #TODO
        
        # Use reliable_qos for publishers
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', reliable_qos)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', reliable_qos)
        #self.state_pub = self.create_publisher(Float32MultiArray, '/vrx/state', 10)
        #self.reward_pub = self.create_publisher(Float64MultiArray, '/vrx/reward', 10)
        self.step_publisher = self.create_publisher(VRXStepData, '/vrx/step_data', reliable_qos)

        
        self.create_timer(0.1, self.control_loop)                                                           # Control logic loop timer 
                                                                                                            #TODO  change if needed
        # Initialise 
        self.gps_data = None
        self.imu_data = None
        self.current_waypoint = None
        #self.current_state = None
        self.last_action = None
        #self.episode_step = 0
        #self.max_steps = 1000
        self.task_state = None

        
    def gps_callback(self, msg):                                                                            # handle incomming GPS data from environment 
        self.gps_data = [msg.latitude, msg.longitude, msg.altitude]
        self.waypoint_manager.update_position(self.gps_data)                                                # update waypoint manager with new info
        
    def imu_callback(self, msg):                                                                            # handles incomming IMU data from environment 
        self.imu_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                                                                                                            #TODO Maybe also send to waypoint manager to update "heading"
        
    def task_info_callback(self, msg):                                                                      # handles current task state
        self.task_state = msg.state                                                                         #TODO implement ROS2 topic for Task
    
    def action_callback(self, msg):                                                                         # Receives Action from DRL Agent 
        self.last_action = msg.data
        
    def get_reward(self):                                                                                  #TODO Implement my own Reward function class 
        return 0.0                                                                                         # Distance to current waypoint, Smoothness of trajectory, Energy efficiency, task completion progress etc.
    

    def control_loop(self):                                                                                 # Main control Loop called by control timer
        if self.gps_data is None or self.imu_data is None:                                                  # checks if GPS and IMU data are available before proceeding (Simulation/Environment began)
            return
        
        self.current_waypoint = self.waypoint_manager.get_current_waypoint()                                # current waypoint

        #move to single custom message
        '''#self.current_state = np.array(self.gps_data + self.imu_data + self.current_waypoint)                # Combines data into single State Array
        #state_msg = Float32MultiArray(data=self.current_state.tolist())
        #elf.state_pub.publish(state_msg)'''                                                                   # publishes state for DRL Agent
        
        if self.last_action is not None:
                                                                                                            # checks for available action
            # Publish thruster commands
            left_thrust_msg = Float64()
            right_thrust_msg = Float64()
            left_thrust_msg.data = self.last_action[0]
            right_thrust_msg.data = self.last_action[1]
            self.left_thruster_pub.publish(left_thrust_msg)                                                 # publishes action to control UASV 
            self.right_thruster_pub.publish(right_thrust_msg)
            
        reward = self.get_reward()
        task_finished = (self.task_state == Task.STATE_FINISHED) #TODO implement task logic

        step_data = VRXStepData()
        step_data.gps_data = self.gps_data
        step_data.imu_data = self.imu_data
        step_data.current_waypoint = self.current_waypoint
        step_data.reward = reward
        step_data.task_finished = task_finished
        
        self.step_publisher.publish(step_data)

        #episode logic moved to DRL Agent 
        '''#done = (self.episode_step >= self.max_steps) or (self.task_state == Task.STATE_FINISHED)            # get reward from reward function
        #reward_msg = Float64MultiArray()
        #reward_msg.data = [reward, 1.0 if done else 0.0]                                                    # Publishes reward For the DRL Agent
        #self.reward_pub.publish(reward_msg)
        #self.episode_step += 1
        #if done:                                                                                            # incrementing episode count + done flag
        #    self.reset_episode()'''
    
    def reset_episode(self):                                                                            #TODO reset logic for reseting environment etc.
        self.episode_step = 0
        self.gps_data = None
        self.imu_data = None
        self.current_state = None
        self.last_action = None
        self.waypoint_manager.reset()
        # Implement logic to reset the VRX simulation   

def main(args=None):
    rclpy.init(args=args)
    controller = VRXController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()