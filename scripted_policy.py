import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from scipy.spatial.transform import Rotation as R

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, left_pose_subscriber, right_pose_subscriber, inject_noise=False):
        self.left_subscriber = left_pose_subscriber
        self.right_subscriber = left_pose_subscriber
        self.curr_left_waypoint = None
        self.curr_right_waypoint = None
        self.step_count = 0
        self.inject_noise = inject_noise
        self.init_left_pose = None
        self.init_right_pose = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def interpolate_call(self, curr_pose, rel_move):
        # Interpolate current pose with relative move
        new_position = curr_pose[:3] + rel_move["position"]
        new_orientation = R.from_quat(curr_pose[3:]) * R.from_quat(rel_move["orientation"])
        return new_position, new_orientation.as_quat()

    def __call__(self, ts):
        if self.step_count == 0:
            self.init_left_pose = np.array(ts.observation['mocap_pose_left'])
            self.init_right_pose = np.array(ts.observation['mocap_pose_right'])
            self.curr_left_waypoint = self.init_left_pose
            self.curr_right_waypoint = self.init_right_pose

        # Get the latest relative positions and orientations
        if self.left_subscriber.latest_relative_position is not None:
            left_rel_move = {
                "position": self.left_subscriber.latest_relative_position,
                "orientation": self.left_subscriber.latest_relative_orientation
            }
            left_xyz, left_quat = self.interpolate_call(self.curr_left_waypoint, left_rel_move)
            self.curr_left_waypoint = np.concatenate([left_xyz, left_quat])
        else:
            left_xyz, left_quat = self.curr_left_waypoint[:3], self.curr_left_waypoint[3:]

        if self.right_subscriber.latest_relative_position is not None:
            right_rel_move = {
                "position": self.right_subscriber.latest_relative_position,
                "orientation": self.right_subscriber.latest_relative_orientation
            }
            right_xyz, right_quat = self.interpolate_call(self.curr_right_waypoint, right_rel_move)
            self.curr_right_waypoint = np.concatenate([right_xyz, right_quat])
        else:
            right_xyz, right_quat = self.curr_right_waypoint[:3], self.curr_right_waypoint[3:]

        left_gripper = 0  # Define gripper state as needed
        right_gripper = 0  # Define gripper state as needed

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])

if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
