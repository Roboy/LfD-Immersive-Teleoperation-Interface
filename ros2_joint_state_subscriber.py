import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self, topic_name='/operator/alice/joint_state'):
        super().__init__('joint_state_subscriber_' + topic_name.split('/')[-1])
        self.subscription = self.create_subscription(JointState, topic_name, self.joint_state_callback, 10)
        self.right_finger_R = None
        self.left_finger_R = None
        self.left_finger_L = None
        self.right_finger_L = None

    def joint_state_callback(self, msg):
        try:
            name_index_map = {name: index for index, name in enumerate(msg.name)}
            self.right_finger_R = msg.position[name_index_map['right_finger_R']]
            self.left_finger_R = msg.position[name_index_map['left_finger_R']]
            self.left_finger_L = msg.position[name_index_map['left_finger_L']]
            self.right_finger_L = msg.position[name_index_map['right_finger_L']]

        except KeyError as e:
            self.get_logger().error(f'Joint name not found: {e}')