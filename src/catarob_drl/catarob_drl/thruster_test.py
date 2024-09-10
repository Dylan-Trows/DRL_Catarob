import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ThrusterTestNode(Node):
    def __init__(self):
        super().__init__('thruster_test_node')
        self.left_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thruster_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        
        self.timer = self.create_timer(10, self.timer_callback)
        self.test_phase = 0

    def timer_callback(self):
        msg = Float64()
        if self.test_phase == 0:
            # Test forward motion
            msg.data = 10.0
            self.left_thruster_pub.publish(msg)
            self.right_thruster_pub.publish(msg)
            self.get_logger().info('Testing forward motion')
        elif self.test_phase == 1:
            # Test backward motion
            msg.data = -10.0
            self.left_thruster_pub.publish(msg)
            self.right_thruster_pub.publish(msg)
            self.get_logger().info('Testing backward motion')
        elif self.test_phase == 2:
            # Test turning left
            msg.data = 0.0
            self.left_thruster_pub.publish(msg)
            msg.data = 10.0
            self.right_thruster_pub.publish(msg)
            self.get_logger().info('Testing left turn')
        elif self.test_phase == 3:
            # Test turning right
            msg.data = 10.0
            self.left_thruster_pub.publish(msg)
            msg.data = 0.0
            self.right_thruster_pub.publish(msg)
            self.get_logger().info('Testing right turn')
        
        else:
            # Stop all thrusters
            msg.data = 0.0
            self.left_thruster_pub.publish(msg)
            self.right_thruster_pub.publish(msg)
            
            self.get_logger().info('Test complete. Stopping all thrusters.')
            self.timer.cancel()

        self.test_phase += 1

def main(args=None):
    rclpy.init(args=args)
    thruster_test_node = ThrusterTestNode()
    rclpy.spin(thruster_test_node)
    thruster_test_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()