import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, PickPolicy
import cv2


import IPython
e = IPython.embed

import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ros2_subscriber import SingletonSubscriber

ros_shutdown_flag = False

class CameraNode(Node):
    def __init__(self):
        super().__init__("stereocamera_node")
        self.left_image_topic_ = self.declare_parameter("left_image_topic", "/image/left/image_compressed").value
        self.right_image_topic_ = self.declare_parameter("right_image_topic", "/image/right/image_compressed").value
        self.exception_counter = 0
        self.fps = 5  # Adjust FPS as necessary
        self.left_frame_id_ = self.declare_parameter("left_frame_id", "left_camera").value
        self.right_frame_id_ = self.declare_parameter("right_frame_id", "right_camera").value
        self.left_image_publisher_ = self.create_publisher(CompressedImage, self.left_image_topic_, 1)
        self.right_image_publisher_ = self.create_publisher(CompressedImage, self.right_image_topic_, 1)
        
        self.br = CvBridge()

        self.timer = self.create_timer(1.0/self.fps, self.image_callback)
        self.get_logger().info(f"Stereocamera node ready with {self.fps} FPS")


    def image_callback(self):
        try:
            time_msg = self.get_time_msg()
            if hasattr(self, 'first_plt_img') and hasattr(self, 'sec_plt_img'):
                # Resize images to a quarter of their original size
                resized_first_img = cv2.resize(self.first_plt_img, (self.first_plt_img.shape[1] // 2, self.first_plt_img.shape[0] // 2))
                resized_sec_img = cv2.resize(self.sec_plt_img, (self.sec_plt_img.shape[1] // 2, self.sec_plt_img.shape[0] // 2))

                # Create black frames
                frame_height = max(self.first_plt_img.shape[0], self.sec_plt_img.shape[0]) * 2
                frame_width = max(self.first_plt_img.shape[1], self.sec_plt_img.shape[1]) * 2
                black_frame_first = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                black_frame_sec = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                # Position resized images
                resized_first_y = (frame_height - resized_first_img.shape[0]) // 2
                resized_first_x = (frame_width - resized_first_img.shape[1]) // 2
                black_frame_first[resized_first_y:resized_first_y + resized_first_img.shape[0], resized_first_x:resized_first_x + resized_first_img.shape[1]] = resized_first_img

                resized_sec_y = (frame_height - resized_sec_img.shape[0]) // 2
                resized_sec_x = (frame_width - resized_sec_img.shape[1]) // 2
                black_frame_sec[resized_sec_y:resized_sec_y + resized_sec_img.shape[0], resized_sec_x:resized_sec_x + resized_sec_img.shape[1]] = resized_sec_img

                # Position original images within the resized images
                original_first_y = resized_first_y + (resized_first_img.shape[0] - self.first_plt_img.shape[0]) // 2
                original_first_x = resized_first_x + (resized_first_img.shape[1] - self.first_plt_img.shape[1]) // 2
                black_frame_first[original_first_y:original_first_y + self.first_plt_img.shape[0], original_first_x:original_first_x + self.first_plt_img.shape[1]] = self.first_plt_img

                original_sec_y = resized_sec_y + (resized_sec_img.shape[0] - self.sec_plt_img.shape[0]) // 2
                original_sec_x = resized_sec_x + (resized_sec_img.shape[1] - self.sec_plt_img.shape[1]) // 2
                black_frame_sec[original_sec_y:original_sec_y + self.sec_plt_img.shape[0], original_sec_x:original_sec_x + self.sec_plt_img.shape[1]] = self.sec_plt_img

                 # Convert frames to image messages
                left_img_msg = self.get_image_msg(black_frame_first, time_msg, compressed=True)
                right_img_msg = self.get_image_msg(black_frame_sec, time_msg, compressed=True, left=False)

                # Publish the image messages
                self.left_image_publisher_.publish(left_img_msg)
                self.right_image_publisher_.publish(right_img_msg)
        except Exception as e:
            # Handle exceptions if needed
            print(f"Error in image_callback: {e}")

    def get_time_msg(self):
        time_msg = self.get_clock().now().to_msg()
        return time_msg

    def get_image_msg(self, image, time, compressed=False, left=True):
        img_msg = self.br.cv2_to_compressed_imgmsg(image, dst_format='jpeg') if compressed else self.br.cv2_to_imgmsg(image)
        img_msg.header.stamp = time
        img_msg.header.frame_id = self.left_frame_id_ if left else self.right_frame_id_
        return img_msg

def ros_spin():
    global ros_shutdown_flag
    rclpy.init()
    subscriber_node = SingletonSubscriber.get_instance()
    camera_node = CameraNode()
    try:
        while rclpy.ok() and not ros_shutdown_flag:
            rclpy.spin_once(subscriber_node, timeout_sec=1.0)
            rclpy.spin_once(camera_node, timeout_sec=1.0)
    finally:
        print("Shutting down ROS node")
        subscriber_node.destroy_node()
        camera_node.destroy_node()
        rclpy.shutdown()
        print("ROS node destroyed and shutdown completed")

def ros_kill():
    global ros_shutdown_flag
    print("Setting shutdown flag")
    ros_shutdown_flag = True

def main(args):
    global ros_shutdown_flag

    ros_thread = threading.Thread(target=ros_spin)
    ros_thread.start()
    subscriber_instance = SingletonSubscriber.get_instance()


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
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_lift_cube_scripted':
        policy_cls = PickPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        # This method resets the environment to its initial state and returns the first TimeStep.
        ts = env.reset(subscriber_instance)

        episode = [ts]
        policy = policy_cls(inject_noise)
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
            ts = env.step(action, subscriber_instance)
            episode.append(ts)
            if onscreen_render:
                first_plt_img.set_data(ts.observation['images'][first_cam])
                sec_plt_img.set_data(ts.observation['images'][second_cam])
                # Update images in CameraNode
                CameraNode.first_plt_img = ts.observation['images'][first_cam]
                CameraNode.sec_plt_img = ts.observation['images'][second_cam]
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
        ts = env.reset(subscriber_instance)

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))  # 1 Zeile, 2 Spalten
            first_plt_img = ax.imshow(ts.observation['images'][first_cam])
            sec_plt_img = bx.imshow(ts.observation['images'][second_cam])
            plt.ion()
        for t in range(len(joint_traj)):  # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action, subscriber_instance)
            episode_replay.append(ts)
            if onscreen_render:
                first_plt_img.set_data(ts.observation['images'][first_cam])
                sec_plt_img.set_data(ts.observation['images'][second_cam])
                # Update images in CameraNode
                CameraNode.first_plt_img = ts.observation['images'][first_cam]
                CameraNode.sec_plt_img = ts.observation['images'][second_cam]
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
