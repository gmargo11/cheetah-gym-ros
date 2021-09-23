#!/usr/bin/env python

#import imutils
import numpy as np
import rospy
from cv_bridge import CvBridge
import cv2

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand, ImageRequest, TrajParams
import sensor_msgs.msg

from cheetahgym.systems.pybullet_system import PyBulletSystem
from cheetahgym.data_types.low_level_types import LowLevelCmd, LowLevelState
from cheetahgym.data_types.mpc_level_types import MPCLevelCmd, MPCLevelState
from cheetahdeploy.evaluation.evaluation_utils import load_cfg, build_env, get_agent
from cheetahgym.data_types.camera_parameters import cameraParameters
from cheetahgym.utils.rotation_utils import get_rotation_matrix_from_quaternion, get_rotation_matrix_from_rpy, inversion



class PolicyServer():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self, cfg=None):
        
        self.hidden_state = None
        self.is_rnn = False
        self.BC = False

        self.low_level_state = LowLevelState()
        self.robot_state_is_initialized = False

        self.pd_torque_pub = rospy.Publisher("pd_torque_command",
            PDPlusTorqueCommand, queue_size=10)

        self.robot_state_sub = rospy.Subscriber("robot_state",
            RobotState, self.robot_state_callback)

        self.params_sub = rospy.Subscriber("traj_params",
            TrajParams, self.params_callback)

    def robot_state_callback(self, robot_state_msg):
        
        self.low_level_state.body_pos = robot_state_msg.body_pos
        self.low_level_state.body_rpy = robot_state_msg.body_rpy
        self.low_level_state.body_linear_vel = robot_state_msg.body_linear_vel
        self.low_level_state.body_angular_vel = robot_state_msg.body_angular_vel
        self.low_level_state.joint_pos = np.array(robot_state_msg.joint_pos)
        self.low_level_state.joint_vel = np.array(robot_state_msg.joint_vel)

        self.robot_state_is_initialized = True

        # convert to correct form for input to policy (WATCH OUT FOR VECTOR NORMALIZATION! USE ob_filt)

        
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


    def infer(self):

        if self.is_rnn:
            if self.BC:
                ret = self.agent.get_action(self.obs, sample=False, hidden_state=hidden_state, evaluation=True)
            else:
                ret = self.agent.get_action(self.obs, sample=False, hidden_state=hidden_state)
        else:
            if self.BC:
                ret = self.agent.get_action(self.obs, sample=False, evaluation=True)
            else:
                ret = self.agent.get_action(self.obs, sample=False)
        if len(ret) > 2:
            self.is_rnn = True
            self.action, self.action_info, self.hidden_state = ret
        else:
            self.action, self.action_info = ret
        



if __name__ == '__main__':
    rospy.init_node('PolicyServer', anonymous=True)
    r = rospy.Rate(500)
    mpc_ctrl = PolicyServer()
    while not rospy.is_shutdown():
        mpc_ctrl.infer()
        r.sleep()
