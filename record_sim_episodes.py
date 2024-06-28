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

from ros2_headset_pos_subscriber import HeadsetSubscriber
from ros2_sim_img_publisher import CameraNode
from ros2_contr_pose_subscriber import PoseSubscriber
from ros2_joint_state_subscriber import JointStateSubscriber


import IPython
e = IPython.embed

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
    dataset_dir = args['dataset_dir']
    required_num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    first_cam = 'left_eye'
    second_cam = 'right_eye'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    successful_episodes = 0

    while True:
        try:
            successful_episodes = int(input(f"How many successful episodes are already saved (0-{required_num_episodes})? "))
            if successful_episodes < 0 or successful_episodes > required_num_episodes:
                raise ValueError
            break
        except ValueError:
            print(f"Invalid input. Please enter a number between 0 and {required_num_episodes}.")


    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    while successful_episodes < required_num_episodes:
        try:
            policy = BasePolicy(shared_state.left_pose_subscriber, shared_state.right_pose_subscriber, shared_state.joint_subscriber, inject_noise)
        except Exception as e:
            print(f"Error initializing policy: {e}")
            return
        

        print(f'Starting episode {successful_episodes + 1}')
        success = False

        while not success:
            print('Rolling out EE space scripted policy')
            # setup the environment
            env = make_ee_sim_env(task_name)
            ts = env.reset(shared_state.headset_subscriber)

            episode = [ts]
            policy = BasePolicy(shared_state.left_pose_subscriber, shared_state.right_pose_subscriber, shared_state.joint_subscriber, inject_noise)

            # setup plotting
            if onscreen_render:
                fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))
                first_plt_img = ax.imshow(ts.observation['images'][first_cam])
                sec_plt_img = bx.imshow(ts.observation['images'][second_cam])
                plt.ion()

            for step in range(episode_len):
                action = policy(ts)
                ts = env.step(action, shared_state.headset_subscriber)
                episode.append(ts)
                if onscreen_render:
                    first_plt_img.set_data(ts.observation['images'][first_cam])
                    sec_plt_img.set_data(ts.observation['images'][second_cam])
                    shared_state.camera_publisher.first_plt_img = ts.observation['images'][first_cam]
                    shared_state.camera_publisher.sec_plt_img = ts.observation['images'][second_cam]
                    plt.pause(0.002)
            plt.close()

            episode_return = np.sum([ts.reward for ts in episode[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode[1:]])
            if episode_max_reward == env.task.max_reward:
                print(f"Episode Successful, {episode_return=}")
            else:
                print(f"Episode Failed")

            joint_traj = [ts.observation['qpos'] for ts in episode]
            gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
            for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                joint[6] = left_ctrl
                joint[6+7] = right_ctrl

            subtask_info = episode[0].observation['env_state'].copy()

            del env
            del episode
            del policy

            print('Replaying joint commands')
            env = make_sim_env(task_name)
            BOX_POSE[0] = subtask_info
            ts = env.reset(shared_state.headset_subscriber)

            episode_replay = [ts]
            if onscreen_render:
                fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))
                first_plt_img = ax.imshow(ts.observation['images'][first_cam])
                sec_plt_img = bx.imshow(ts.observation['images'][second_cam])
                plt.ion()
            for t in range(len(joint_traj)):
                action = joint_traj[t]
                ts = env.step(action, shared_state.headset_subscriber)
                episode_replay.append(ts)
                if onscreen_render:
                    first_plt_img.set_data(ts.observation['images'][first_cam])
                    sec_plt_img.set_data(ts.observation['images'][second_cam])
                    shared_state.camera_publisher.first_plt_img = ts.observation['images'][first_cam]
                    shared_state.camera_publisher.sec_plt_img = ts.observation['images'][second_cam]
                    plt.pause(0.002)

            episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
            if episode_max_reward == env.task.max_reward:
                print(f"Episode Replay Successful, {episode_return=}")
                success = True
            else:
                print(f"Episode Replay Failed")

            plt.close()

            user_input = input("Do you want to save this episode? (yes/no): ").strip().lower()
            if user_input != 'yes':
                success = False
                continue

            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

            joint_traj = joint_traj[:-1]
            episode_replay = episode_replay[:-1]

            max_timesteps = len(joint_traj)
            while joint_traj:
                action = joint_traj.pop(0)
                ts = episode_replay.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{successful_episodes}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                                chunks=(1, 480, 640, 3))
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
            print("Number of successful episodes: " + str(successful_episodes))

            successful_episodes += 1

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
