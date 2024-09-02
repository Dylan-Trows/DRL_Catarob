import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from vrx_gazebo.msg import Task

class VRXTD3Controller(Node):
    def __init__(self):
        super().__init__('vrx_td3_controller')
        
        # Initialize TD3 algorithm
        self.td3 = TD3(state_dim, action_dim, max_action)
        
        # Subscribe to sensor data from VRX
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        self.create_subscription(Float32, '/wamv/sensors/cameras/front_left_camera/image/compressed/theora/parameter_descriptions', self.camera_callback, 10)
        self.create_subscription(Task, '/vrx/task/info', self.task_info_callback, 10)
        
        # Publisher for control commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/wamv/cmd_vel', 10)
        
        # Timer for control loop
        self.create_timer(0.1, self.control_loop)
        
        # Initialize state variables
        self.current_state = None
        self.episode_step = 0
        self.max_steps = 1000  # Max steps per episode
        
    def gps_callback(self, msg):
        # Process GPS data
        self.gps_data = [msg.latitude, msg.longitude, msg.altitude]
        
    def imu_callback(self, msg):
        # Process IMU data
        self.imu_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        
    def camera_callback(self, msg):
        # Process camera data if needed
        pass
        
    def task_info_callback(self, msg):
        # Process task information
        self.task_state = msg.state
        
    def control_loop(self):
        if self.gps_data is None or self.imu_data is None:
            return
        
        # Combine sensor data to form the state
        self.current_state = np.array(self.gps_data + self.imu_data)
        
        # Get action from TD3
        action = self.td3.select_action(self.current_state)
        
        # Apply action to the robot
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]  # Forward velocity
        cmd_vel.angular.z = action[1]  # Angular velocity
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Check if episode is done
        self.episode_step += 1
        if self.episode_step >= self.max_steps or self.task_state == Task.STATE_FINISHED:
            self.reset_simulation()
        
    def reset_simulation(self):
        # In VRX, resetting might involve waiting for the next task to start
        # You may need to implement a custom service or use existing VRX services for resetting
        self.episode_step = 0

def main(args=None):
    rclpy.init(args=args)
    controller = VRXTD3Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()