#!/usr/bin/env python

#import imutils
import numpy as np
import rospy
from cv_bridge import CvBridge
import cv2
import time
import copy

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand, ImageRequest, TrajParams, TerrainHeightmap
import sensor_msgs.msg

from cheetahgym.systems.pybullet_system import PyBulletSystem
from cheetahgym.data_types.low_level_types import LowLevelCmd, LowLevelState
from cheetahgym.data_types.mpc_level_types import MPCLevelCmd, MPCLevelState
from cheetahdeploy.evaluation.evaluation_utils import load_cfg, build_env, get_agent
from cheetahgym.data_types.camera_parameters import cameraParameters
from cheetahgym.utils.rotation_utils import get_rotation_matrix_from_quaternion, get_rotation_matrix_from_rpy, inversion



class PolicyServer():
    """
    A class for performing inference with a learned gait policy.
    Subscribes to: /robot_state (RobotState) : the body and joint state of the robot.
    Subscribes to: /camera_image (Image) : the depth image from the onboard camera.
    Publishes to: /traj_params (TrajParams) : the parameters of the desired robot trajectory.
    """
    def __init__(self, cfg=None):
        
        self.hidden_state = None
        self.is_rnn = False
        self.BC = False

        self.low_level_state = LowLevelState()
        self.camera_image = np.zeros((1, 15, 48))
        self.prev_mpc_action = MPCLevelCmd()
        self.mpc_progress = 0
        self.action = np.zeros((1, 28))
        self.robot_state_is_initialized = False

        self.load_policy(rospy.get_param('model_path'))

        self.robot_state_sub = rospy.Subscriber("robot_state",
            RobotState, self.robot_state_callback)\

        self.camera_sub = rospy.Subscriber("camera_image",
            sensor_msgs.msg.Image, self.camera_image_callback)

        self.local_map_sub = rospy.Subscriber("local_heightfield",
            TerrainHeightmap, self.local_heightfield_callback)

        self.traj_params_pub = rospy.Publisher("traj_params",
            TrajParams, queue_size=10)


    def robot_state_callback(self, robot_state_msg):
        
        self.low_level_state.body_pos = robot_state_msg.body_pos
        self.low_level_state.body_rpy = robot_state_msg.body_rpy
        self.low_level_state.body_linear_vel = robot_state_msg.body_linear_vel
        self.low_level_state.body_angular_vel = robot_state_msg.body_angular_vel
        self.low_level_state.joint_pos = np.array(robot_state_msg.joint_pos)
        self.low_level_state.joint_vel = np.array(robot_state_msg.joint_vel)

        self.robot_state_is_initialized = True

        # convert to correct form for input to policy (WATCH OUT FOR VECTOR NORMALIZATION! USE ob_filt)

    def camera_image_callback(self, camera_image_msg):

        self.camera_image = camera_image_msg

    def local_heightfield_callback(self, local_heightfield_msg):

        self.camera_image = local_heightfield_msg.heightfield
        
    def publish_traj_params(self):
        if not self.robot_state_is_initialized:
            return

        traj_params_msg = TrajParams()
        traj_params_msg.vel_cmd = self.prev_mpc_action.vel_cmd.tolist()
        traj_params_msg.vel_rpy_cmd = self.prev_mpc_action.vel_rpy_cmd.tolist()
        traj_params_msg.fp_rel_cmd = self.prev_mpc_action.fp_rel_cmd.tolist()
        traj_params_msg.fh_rel_cmd = self.prev_mpc_action.fh_rel_cmd.tolist()
        traj_params_msg.footswing_height = self.prev_mpc_action.footswing_height
        traj_params_msg.vel_table_update = self.prev_mpc_action.vel_table_update.flatten().tolist()
        traj_params_msg.vel_rpy_table_update = self.prev_mpc_action.vel_rpy_table_update.flatten().tolist()
        traj_params_msg.iterations_table_update = self.prev_mpc_action.iterations_table_update.astype(int).flatten().tolist()
        traj_params_msg.planningHorizon = self.prev_mpc_action.planningHorizon
        traj_params_msg.adaptationHorizon = self.prev_mpc_action.adaptationHorizon
        traj_params_msg.adaptationSteps = self.prev_mpc_action.adaptationSteps
        traj_params_msg.iterationsBetweenMPC = int(self.prev_mpc_action.iterationsBetweenMPC)
        traj_params_msg.offsets_smoothed = self.prev_mpc_action.offsets_smoothed.astype(int).tolist()
        traj_params_msg.durations_smoothed = self.prev_mpc_action.durations_smoothed.astype(int).tolist()
        traj_params_msg.mpc_table_update = self.prev_mpc_action.mpc_table_update.astype(int).flatten().tolist()
        
        self.traj_params_pub.publish(traj_params_msg)

    def load_policy(self, model_path):

        cfg = load_cfg(model_path)
        cfg.alg.num_envs = 1

        # SETTINGS
        cfg.alg.device = "cuda"
        cfg.alg.randomize_dynamics = False
        cfg.alg.randomize_contact_dynamics = False
        cfg.alg.external_force_magnitude = 0.0
        cfg.alg.external_torque_magnitude = 0.0
        cfg.alg.datetime = time.strftime("%Y%m%d-%H%M%S")

        # ## red observation noise
        # cfg.alg.height_std = 0.04
        # cfg.alg.roll_std = 0.04
        # cfg.alg.pitch_std = 0.12
        # cfg.alg.yaw_std = 0.04
        # cfg.alg.vel_std = 0.3
        # cfg.alg.vel_roll_std = 0.3
        # cfg.alg.vel_pitch_std = 0.8
        # cfg.alg.vel_yaw_std = 0.3
        # #cfg.alg.cam_pose_std = 0.03
        # #cfg.alg.cam_rpy_std = 0.2
        # cfg.alg.ob_noise_autocorrelation = 0.9


        self.env = build_env(env_name=cfg.alg.env_name, terrain_cfg_file=cfg.alg.terrain_cfg_file, cfg=cfg)
        self.env.reset()
        self.agent = get_agent(env=self.env, cfg=cfg)

        if cfg.alg.expert_save_dir != "None":
            self.BC = True


    def get_env_obs(self, robot_state, camera_image, prev_action, prev_action_raw, mpc_progress):
        ob_scaled_vis = self.env.build_ob_scaled_vis_from_state(robot_state, camera_image, prev_action, prev_action_raw, mpc_progress)

        if callable(getattr(self.env, "_obfilt", None)):
            ob_scaled_vis = self.env._obfilt(ob_scaled_vis)

        return ob_scaled_vis


    def infer(self):
        obs = self.get_env_obs(self.low_level_state, self.camera_image, self.prev_mpc_action, self.action[0], self.mpc_progress)

        if self.is_rnn:
            if self.BC:
                ret = self.agent.get_action(obs, sample=False, hidden_state=self.hidden_state, evaluation=True)
            else:
                ret = self.agent.get_action(obs, sample=False, hidden_state=self.hidden_state)
        else:
            if self.BC:
                ret = self.agent.get_action(obs, sample=False, evaluation=True)
            else:
                ret = self.agent.get_action(obs, sample=False)
        if len(ret) > 2:
            self.is_rnn = True
            self.action, self.action_info, self.hidden_state = ret
        else:
            self.action, self.action_info = ret


        self.env.update_cmd(self.action[0])
        self.prev_mpc_action = copy.deepcopy(self.env.mpc_level_cmd)

        self.mpc_progress += 1

        self.publish_traj_params()



if __name__ == '__main__':
    rospy.init_node('PolicyServer', anonymous=True)
    r = rospy.Rate(500)
    policy_server = PolicyServer()
    while not rospy.is_shutdown():
        policy_server.infer()
        r.sleep()
