import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from scipy.interpolate import CubicSpline

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from scipy.spatial.transform import Rotation as R

import IPython
e = IPython.embed

class BasePolicy:
    def __init__(self, left_pose_subscriber, right_pose_subscriber, joint_subscriber, inject_noise=False):
        self.left_subscriber = left_pose_subscriber
        self.right_subscriber = right_pose_subscriber
        self.joint_subscriber = joint_subscriber
        self.curr_left_waypoint = None
        self.curr_right_waypoint = None
        self.step_count = 0
        self.inject_noise = inject_noise
        self.init_left_pose = None
        self.init_right_pose = None
        self.left_trajectory = None
        self.right_trajectory = None
        self.left_rel_move = None
        self.right_rel_move = None

    @staticmethod
    def interpolate(waypoints, t):
        times = np.linspace(0, 1, len(waypoints))
        positions = np.array([wp["xyz"] for wp in waypoints])
        orientations = np.array([wp["quat"] for wp in waypoints])
        grippers = np.array([wp["gripper"] for wp in waypoints])

        pos_spline = CubicSpline(times, positions, axis=0)
        quat_spline = CubicSpline(times, orientations, axis=0)
        grip_spline = CubicSpline(times, grippers, axis=0)

        return pos_spline(t), quat_spline(t), grip_spline(t)

    def plan_trajectory(self, curr_waypoint, next_waypoint, steps=10):
        waypoints = [
            {"t": 0, "xyz": curr_waypoint[:3], "quat": curr_waypoint[3:7], "gripper": curr_waypoint[7]},
            {"t": 1, "xyz": next_waypoint[:3], "quat": next_waypoint[3:7], "gripper": next_waypoint[7]}
        ]
        times = np.linspace(0, 1, steps)
        trajectory = [self.interpolate(waypoints, t) for t in times]
        return trajectory

    def interpolate_call(self, curr_pose, rel_move):
        converted_position = np.array([
            rel_move["position"][0],  # +x
            rel_move["position"][1],  # +y
            rel_move["position"][2]   # +z
        ])
        new_position = curr_pose[:3] + converted_position
        new_orientation = R.from_quat(curr_pose[3:7]) * R.from_quat(rel_move["orientation"])
        return new_position, new_orientation.as_quat()

    def get_gripper_state(self):
        # Define a small threshold to consider the gripper as open
        threshold = 1e-5  # Adjust this value as necessary based on observed data

        # Determine gripper state based on joint subscriber values and threshold
        left_gripper_open = (abs(self.joint_subscriber.left_finger_L) < threshold and
                            abs(self.joint_subscriber.right_finger_L) < threshold)
        right_gripper_open = (abs(self.joint_subscriber.left_finger_R) < threshold and
                            abs(self.joint_subscriber.right_finger_R) < threshold)

        left_gripper = 1 if left_gripper_open else 0
        right_gripper = 1 if right_gripper_open else 0

        return left_gripper, right_gripper


    def __call__(self, ts):
        if self.step_count == 0:
            self.init_left_pose = np.array(ts.observation['mocap_pose_left'])
            self.init_right_pose = np.array(ts.observation['mocap_pose_right'])
            self.curr_left_waypoint = np.concatenate([self.init_left_pose, [0]])  # Assuming initial gripper state is 0
            self.curr_right_waypoint = np.concatenate([self.init_right_pose, [0]])  # Assuming initial gripper state is 0

        if self.left_subscriber.latest_relative_position is not None:
            self.left_rel_move = {
                "position": self.left_subscriber.latest_relative_position,
                "orientation": self.left_subscriber.latest_relative_orientation
            }

            left_xyz, left_quat = self.interpolate_call(self.curr_left_waypoint, self.left_rel_move)
            left_gripper, _ = self.get_gripper_state()
            self.curr_left_waypoint = np.concatenate([left_xyz, left_quat, [left_gripper]])  # Update gripper state
            self.left_trajectory = self.plan_trajectory(self.curr_left_waypoint, self.curr_left_waypoint)
        else:
            left_xyz, left_quat = self.curr_left_waypoint[:3], self.curr_left_waypoint[3:7]
            left_gripper = self.curr_left_waypoint[7]

        if self.right_subscriber.latest_relative_position is not None:
            self.right_rel_move = {
                "position": self.right_subscriber.latest_relative_position,
                "orientation": self.right_subscriber.latest_relative_orientation
            }

            right_xyz, right_quat = self.interpolate_call(self.curr_right_waypoint, self.right_rel_move)
            _, right_gripper = self.get_gripper_state()
            self.curr_right_waypoint = np.concatenate([right_xyz, right_quat, [right_gripper]])  # Update gripper state
            self.right_trajectory = self.plan_trajectory(self.curr_right_waypoint, self.curr_right_waypoint)
        else:
            right_xyz, right_quat = self.curr_right_waypoint[:3], self.curr_right_waypoint[3:7]
            right_gripper = self.curr_right_waypoint[7]

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
