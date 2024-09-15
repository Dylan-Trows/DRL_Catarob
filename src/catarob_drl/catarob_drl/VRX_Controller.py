import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float32MultiArray, Float64MultiArray
import math
import numpy as np
from .Waypoint_manager import WaypointManager            
from catarob_drl_interfaces.msg import VRXStepData 

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
        #TODO Add Example Waypoints here !
        self.waypoint_manager.add_waypoint(-33.722499646, 150.674243934, 1.161502034)                                  # Example waypoint in VRX

        # Use sensor_qos for sensor data subscriptions
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, sensor_qos)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, sensor_qos)
        
        # Use reliable_qos for task info and action subscriptions
        #self.create_subscription(Task, '/vrx/task/info', self.task_info_callback, reliable_qos)                       
        self.create_subscription(Float64MultiArray, '/vrx/action', self.action_callback, reliable_qos)                #TODO
        
        # Use reliable_qos for publishers
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', reliable_qos)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', reliable_qos)
        self.step_publisher = self.create_publisher(VRXStepData, '/vrx/step_data', reliable_qos)

        
        self.create_timer(0.1, self.control_loop)                                                           # Control logic loop timer 
                                                                                                            #TODO  change loop frquency if needed
        # Initialise 
        self.current_heading = None
        self.gps_data = None
        self.imu_data = None
        self.current_waypoint = None
        #self.current_state = None
        self.last_action = None
        self.task_finished = False
        self.heading_error = 0.0
        self.previous_distance = None
        self.maximum_distance = None
        self.alpha = 1
        self.beta = 1
        self.max_speed = 2

        
    def gps_callback(self, msg):                                                                            # handle incomming GPS data from environment 
        self.gps_data = [msg.latitude, msg.longitude, msg.altitude]

        print("lat: ",msg.latitude," lon: ",msg.longitude," alt: ", msg.altitude)                           # Test print statement

        self.waypoint_manager.update_position(msg.latitude, msg.longitude, msg.altitude)                    # update waypoint manager with new info
        current_waypoint = self.waypoint_manager.get_current_waypoint()                                     # Get the current waypoint
        if current_waypoint:

            if self.maximum_distance is None and self.previous_distance is None:
                self.maximum_distance = self.waypoint_manager.calculate_distance(msg.latitude, msg.longitude, current_waypoint.latitude, current_waypoint.longitude)            # calculating maximum and previous distance to target
                self.previous_distance = self.maximum_distance
            elif self.maximum_distance is not None:
                self.previous_distance = self.waypoint_manager.calculate_distance(msg.latitude, msg.longitude, current_waypoint.latitude, current_waypoint.longitude)

            desired_heading = self.waypoint_manager.get_desired_heading(msg.latitude, msg.longitude)
            print("desired heading : ", desired_heading)
            if self.current_heading is not None:
                print("current heading : ", self.current_heading)
                # Calculate heading error
                self.heading_error = self.waypoint_manager.calculate_heading_error(self.current_heading, desired_heading)       # Get Heading error for Reward function

        # Check if we've reached all waypoints
        if not self.waypoint_manager.has_more_waypoints():
            self.task_finished = True                                                                       # means that we have reached the Waypoint and the Episode is over !
            # Success!

    def imu_callback(self, msg):                                                                            # handles incomming IMU data from environment 
        self.imu_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        # Extract heading from quaternion
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w             #TODO DECIDE Angular velocity, Linear Acceleration
        self.current_heading = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self.current_heading = (self.current_heading + 360) % 360                                                                                                    
        
    def task_info_callback(self, msg):                                                                      # handles current task state
        self.task_state = msg.state                                                                         #TODO implement ROS2 topic for Task (DONE)
    
    def action_callback(self, msg):                                                                         # Receives Action from DRL Agent 
        self.last_action = msg.data
        
    def get_reward(self):                                                                                    
        reward = 0.0
        if self.current_waypoint is None or self.gps_data is None or self.last_action is None:
            return 0.0

        # 1. Distance Reward 
            
        # Calculate arc length
        l_arc = self.previous_distance * math.radians(abs(self.heading_error))

        # Normalize distances
        normalized_d_target = self.previous_distance / self.maximum_distance
        normalized_l_arc = l_arc / (math.pi * self.previous_distance)

        # Weights for arc and distance to target
        w_arc = 0.5
        w_dist = 0.5

        # Calculate weighted distance
        distance = w_arc * normalized_l_arc + w_dist * normalized_d_target                                  # Incorporating Heading Error and distance to target into reward.

        distance_reward = -self.alpha * math.tanh(distance - 1)
        reward += distance_reward                                                                            

        # 2. Velocity reward
         
        forward_speed = self.last_action[0]                                                                 # get_forward_speed()
        speed_heading_reward = (forward_speed * math.cos(math.radians(self.heading_error))) / self.max_speed
        
        # Option 1: Use the speed-heading reward directly
        movement_reward = self.beta * speed_heading_reward

        # Option 2: Apply the 1-e^(-v) function to shape the reward
        # v = abs(speed_heading_reward)  # Use absolute value to handle negative rewards
        # movement_reward = self.beta * (1 - math.exp(-v))
        # if speed_heading_reward < 0:
        #     movement_reward = -movement_reward  # Preserve the sign

        # Combine rewards 
        reward += movement_reward


        # 3. Penalty for energy usage (assuming self.last_action contains thruster values)                  # TODO Think of logic for energy consumption / smoothness of trajectories
        if self.last_action is not None:
            energy_usage = np.sum(np.abs(self.last_action))
            energy_penalty = -0.1 * energy_usage  # Small penalty for energy usage
            reward += energy_penalty
        if self.task_finished == True :
            reward += 100
        
        return reward                                                                                       # Distance to current waypoint, Smoothness of trajectory, Energy efficiency, task completion progress etc.
    

    def control_loop(self):                                                                                 # Main control Loop called by control timer
        if self.gps_data is None or self.imu_data is None:                                                  # checks if GPS and IMU data are available before proceeding (Simulation/Environment began)
            return
        
        self.current_waypoint = self.waypoint_manager.get_current_waypoint()                                # current waypoint
        
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
        
        if ((self.gps_data != None) & (self.imu_data != None) & (self.current_waypoint != None)):

            step_data = VRXStepData()
            step_data.gps_data = self.gps_data
            step_data.imu_data = self.imu_data
            step_data.current_waypoint = self.current_waypoint
            step_data.reward = reward
            print("Reward: ", reward)
            step_data.task_finished = self.task_finished
            if self.task_finished == True:
                print("task finished!")

            step_data.heading_error = self.heading_error
            step_data.current_heading = self.current_heading
        
            self.step_publisher.publish(step_data)
    
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