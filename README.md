# Learning from Demonstration with VR as an immersive Teleoperation Interface

This repository is build upon the work behind the [ACT-PLUS-PLUS](https://github.com/MarkFzp/act-plus-plus) repository. It enables the user to perform demonstrations for two seperate task setups, which are collected and can be later on used for training a policy, with either ACT or Diffusion Policy. Once training is completed, the learned policy can be evaluated for different rollouts of the task setup.

It should be noted that the VR App used in order to collect the VR headset and controller parameters, was developed by Devanthro. While recording the data demonstrations, a ROS2 connection is established between the main computer running this repository and the Oculus Meta Quest 2 which runs the VR App. This enables bidirectional communication and with this, the movements can be conveyed to the created MuJoCo simulation and the images, derived from the virtual cameras within MuJoCo, are transmitted to the VR Display, enabling the demonstrator to receive visual feedback.

### Repo Structure

- ``assets\bimanual_viperx_ee_transfer_cube.xml`` Task Definitions for single Shape on a table
- ``assets\bimanual_viperx_ee_insertion.xml`` Task Definitions for Shape on a table and a Box next to it
- ``assets\vx300s_(left/right).xml`` Definitions of the left / right Robot Arm


- ``record_sim_episodes.py`` Collect Data Demonstration Episodes
- ``imitate_episodes.py`` Train and Evaluate ACT or Diffusion Policy
- ``scripted_episodes.py`` Apply Data from ROS2 Subscribers to the Simulation
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

- ``ros2_contr_pos_subscriber.py`` ROS2 Subscriber for VR Controller data
- ``ros2_headset_pos_subscriber.py`` ROS2 Subscriber for VR Headset data (can be turned off through the Global Variable in record_sim_episodes.py)
- ``ros2_joint_state_subscriber.py`` ROS2 Subscriber for Gripper Button on the VR Controller
- ``ros2_sim_img_publisher.py`` ROS2 Publisher for virtual Camera Images from the Simulation



### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

- also need to install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch) for Diffusion Policy by `pip install -e .`

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments (LEGACY table-top ALOHA environments)

I use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.

In order to make a connection to the VR App work, install ROS2 (I used Humble Hawksbill) and run:

    ros2 run ros_tcp_endpoint default_server_endpoint
    --ros-args -p ROS_IP:=<your IP address>

on the computer, you want to run this repo on. After connecting to the VR App, you can run the following command in order to generate the amount of data episodes you want to acquire:

    python3 record_sim_episodes.py --task_name <task_name> --dataset_dir
    <path_to_directory> --num_episodes <number_of_episodes>

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the simulated episodes after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT, the following command is recommended:
    
    python3 imitate_episodes.py --task_name <task_name>
    --ckpt_dir <path_to_directory> --policy_class ACT
    --kl_weight 10 --chunk_size 100 --hidden_dim 512
    --batch_size 8 --dim_feedforward 3200 --lr 1e-5
    --seed 0 --num_steps 100000

To train Diffusion Policy, the following command is recommended:

    python3 imitate_episodes.py --task_name <task_name>
    --ckpt_dir <path_to_directory> --policy_class Diffusion
    --chunk_size 32 --batch_size 32 --lr 1e-4 --seed 0 --num_steps 100000
    --eval_every 100000 --validate_every 10000 --save_every 10000

you can either choose "sim_transfer_cube_scripted" or sim_insertion_scripted" for <task_name>, to generate the required setup. 


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.

Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

If your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.
