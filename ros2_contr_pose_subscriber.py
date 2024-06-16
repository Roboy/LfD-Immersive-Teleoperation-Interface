from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseSubscriber(Node):
    def __init__(self, topic_name):
        super().__init__('pose_subscriber_' + topic_name.split('/')[-1])
        self.subscription = self.create_subscription(Pose, topic_name, self.pose_callback, 1)
        self.prev_pose = None
        self.relative_trajectory = []
        self.latest_relative_position = None
        self.latest_relative_orientation = None

    def pose_callback(self, msg):
        current_pose = np.array([msg.position.x, msg.position.y, msg.position.z,
                                 msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        if self.prev_pose is None:
            self.prev_pose = current_pose
        else:
            relative_position = current_pose[:3] - self.prev_pose[:3]
            current_orientation = R.from_quat(current_pose[3:])
            prev_orientation = R.from_quat(self.prev_pose[3:])
            relative_orientation = current_orientation * prev_orientation.inv()
            self.latest_relative_position = relative_position
            self.latest_relative_orientation = relative_orientation.as_quat()
            self.relative_trajectory.append({
                "position": relative_position,
                "orientation": relative_orientation.as_quat()
            })
            self.prev_pose = current_pose
