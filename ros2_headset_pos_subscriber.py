# Importiere ROS2 Pakete in deinem Skript
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class HeadsetSubscriber(Node):
    _instance = None

    @staticmethod
    def get_instance():
        if HeadsetSubscriber._instance is None:
            HeadsetSubscriber()
        return HeadsetSubscriber._instance

    def __init__(self, head_pos_rotating):
        if HeadsetSubscriber._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            super().__init__('singleton_subscriber')
            HeadsetSubscriber._instance = self
            self.subscription = self.create_subscription(
                PoseStamped,
                '/operator/alice/headset_orientation',
                self.listener_callback,
                10)
            self.subscription  # prevent unused variable warning
            self.latest_message = None
            self.head_pos_rotating = head_pos_rotating

    def listener_callback(self, msg):
        self.latest_message = msg
        #self.get_logger().info('Received: "%s"' % msg)

    def get_latest_message(self):
        return self.latest_message

def main(args=None):
    rclpy.init(args=args)
    subscriber_node = HeadsetSubscriber.get_instance()
    rclpy.spin(subscriber_node)
    subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
