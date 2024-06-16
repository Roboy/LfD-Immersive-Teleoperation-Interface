import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from scripted_policy import BasePolicy

from ros2_headset_pos_subscriber import HeadsetSubscriber
from ros2_sim_img_publisher import CameraNode
from ros2_contr_pose_subscriber import PoseSubscriber
from ros2_joint_state_subscriber import JointStateSubscriber

import IPython
import threading
import rclpy

ros_shutdown_flag = False
head_pos_rotating = False

# Initialize shared state
class SharedState:
    def __init__(self):
        self.left_pose_subscriber = None
        self.right_pose_subscriber = None
        self.headset_subscriber = None
        self.camera_publisher = None
        self.joint_subscriber = None
        self.initialized = threading.Event()

shared_state = SharedState()
ros_shutdown_flag = False

# Define the ros_spin function
def ros_spin(shared_state):
    global ros_shutdown_flag
    rclpy.init()
    shared_state.camera_publisher = CameraNode()
    shared_state.headset_subscriber = HeadsetSubscriber(head_pos_rotating)
    shared_state.left_pose_subscriber = PoseSubscriber('/operator/device/controller/left/pose')
    shared_state.right_pose_subscriber = PoseSubscriber('/operator/device/controller/right/pose')
    shared_state.joint_subscriber = JointStateSubscriber()
    shared_state.initialized.set()  # Signal that the subscribers are initialized

    try:
        while rclpy.ok() and not ros_shutdown_flag:
            rclpy.spin_once(shared_state.camera_publisher, timeout_sec=1.0)
            rclpy.spin_once(shared_state.headset_subscriber, timeout_sec=1.0)
            rclpy.spin_once(shared_state.left_pose_subscriber, timeout_sec=1.0)
            rclpy.spin_once(shared_state.right_pose_subscriber, timeout_sec=1.0)
            rclpy.spin_once(shared_state.joint_subscriber, timeout_sec=1.0)
    finally:
        print("Shutting down ROS node")
        shared_state.camera_publisher.destroy_node()
        shared_state.headset_subscriber.destroy_node()
        shared_state.left_pose_subscriber.destroy_node()
        shared_state.right_pose_subscriber.destroy_node()
        shared_state.joint_subscriber.destroy_node()
        rclpy.shutdown()
        print("ROS node destroyed and shutdown completed")

# Define the ros_kill function
def ros_kill():
    global ros_shutdown_flag
    print("Setting shutdown flag")
    ros_shutdown_flag = True

# Define the main function
def main(args):
    global ros_shutdown_flag
    shared_state = SharedState()

    # Start ROS spin in a separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(shared_state,))
    ros_thread.start()

    # Wait until the subscribers are initialized
    shared_state.initialized.wait()

    # Access the pose subscribers
    print("Camera Publisher:", shared_state.camera_publisher)
    print("Headset Subscriber:", shared_state.headset_subscriber)

    task_name = args['task_name']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    first_cam = 'left_eye'
    second_cam = 'right_eye'

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    try:
        policy = BasePolicy(shared_state.left_pose_subscriber, shared_state.right_pose_subscriber, shared_state.joint_subscriber, inject_noise)
    except Exception as e:
        print(f"Error initializing policy: {e}")
        return

    while True:
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        # This method resets the environment to its initial state and returns the first TimeStep.
        ts = env.reset(shared_state.headset_subscriber)

        episode = [ts]
        policy = BasePolicy(shared_state.left_pose_subscriber, shared_state.right_pose_subscriber, shared_state.joint_subscriber, inject_noise)

        # setup plotting
        if onscreen_render:
            fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))  # 1 Zeile, 2 Spalten
            first_plt_img = ax.imshow(ts.observation['images'][first_cam])
            sec_plt_img = bx.imshow(ts.observation['images'][second_cam])
            plt.ion()

        for step in range(1500):
            action = policy(ts)
            ts = env.step(action, shared_state.headset_subscriber)
            episode.append(ts)
            if onscreen_render:
                first_plt_img.set_data(ts.observation['images'][first_cam])
                sec_plt_img.set_data(ts.observation['images'][second_cam])
                # Update images in CameraNode
                shared_state.camera_publisher.first_plt_img = ts.observation['images'][first_cam]
                shared_state.camera_publisher.sec_plt_img = ts.observation['images'][second_cam]
                plt.pause(0.002)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--onscreen_render', action='store_true')

    main(vars(parser.parse_args()))
