import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import BasePolicy
import cv2

from ros2_headset_pos_subscriber import HeadsetSubscriber
from ros2_sim_img_publisher import CameraNode
from ros2_contr_pose_subscriber import PoseSubscriber
from ros2_joint_state_subscriber import JointStateSubscriber


import IPython
e = IPython.embed

import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

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

    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    first_cam = 'left_eye'
    second_cam = 'right_eye'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    try:
        policy = BasePolicy(shared_state.left_pose_subscriber, shared_state.right_pose_subscriber, shared_state.joint_subscriber, inject_noise)
    except Exception as e:
        print(f"Error initializing policy: {e}")
        return

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
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

        '''
        Here, a new timestep is generated on each iteration of the loop by calling env.step(action). 
        The result (ts) is appended to the episode list.
        '''

        for step in range(episode_len):
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

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset(shared_state.headset_subscriber)

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))  # 1 Zeile, 2 Spalten
            first_plt_img = ax.imshow(ts.observation['images'][first_cam])
            sec_plt_img = bx.imshow(ts.observation['images'][second_cam])
            plt.ion()
        for t in range(len(joint_traj)):  # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action, shared_state.headset_subscriber)
            episode_replay.append(ts)
            if onscreen_render:
                first_plt_img.set_data(ts.observation['images'][first_cam])
                sec_plt_img.set_data(ts.observation['images'][second_cam])
                # Update images in CameraNode
                shared_state.camera_publisher.first_plt_img = ts.observation['images'][first_cam]
                shared_state.camera_publisher.sec_plt_img = ts.observation['images'][second_cam]
                plt.pause(0.002)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

    ros_kill()  # Signal the ROS node to shut down
    ros_thread.join()  # Wait for the ROS thread to terminate
    print("ROS thread finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')

    main(vars(parser.parse_args()))
