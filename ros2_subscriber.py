# Importiere ROS2 Pakete in deinem Skript
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class SingletonSubscriber(Node):
    _instance = None

    @staticmethod
    def get_instance():
        if SingletonSubscriber._instance is None:
            SingletonSubscriber()
        return SingletonSubscriber._instance

    def __init__(self):
        if SingletonSubscriber._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            super().__init__('singleton_subscriber')
            SingletonSubscriber._instance = self
            self.subscription = self.create_subscription(
                PoseStamped,
                '/operator/alice/headset_orientation',
                self.listener_callback,
                10)
            self.subscription  # prevent unused variable warning
            self.latest_message = None

    def listener_callback(self, msg):
        self.latest_message = msg
        #self.get_logger().info('Received: "%s"' % msg)

    def get_latest_message(self):
        return self.latest_message

def main(args=None):
    rclpy.init(args=args)
    subscriber_node = SingletonSubscriber.get_instance()
    rclpy.spin(subscriber_node)
    subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
