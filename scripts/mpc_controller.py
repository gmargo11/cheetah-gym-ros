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
from cheetahdeploy.evaluation.evaluation_utils import load_cfg
from cheetahgym.data_types.camera_parameters import cameraParameters
from cheetahgym.utils.rotation_utils import get_rotation_matrix_from_quaternion, get_rotation_matrix_from_rpy, inversion



class MPCController():
    """
    A class for tracking a parameterized gait using model-predictive control.
    Subscribes to: /robot_state (RobotState) : the body and joint state of the robot.
    Subscribes to: /traj_params (TrajParams) : the parameters of the desired robot trajectory.
    Publishes to: /pd_torque_command (PDPlusTorqueCommand) : the commanded pd targets, gains, and feedforward torque.
    """
    def __init__(self):
        
        cfg = load_cfg(rospy.get_param('model_path')).alg

        from cheetahgym.utils.mpc_wbc_bindings import RSCheetahControllerV1
        self.mpc_controller = RSCheetahControllerV1(robot_filename="/workspace/cheetah-gym/cheetahgym/config/quadruped-parameters.yaml",
                                        estimator_filename="/workspace/cheetah-gym/cheetahgym/config/quadruped-estimator-parameters.yaml",
                                        cfg=cfg)
        self.mpc_controller.reset_ctrl()

        if cfg.simulator_name == "PYBULLET":
            self.mpc_controller.reverse_legs = True

        # intialize state
        self.mpc_level_cmd = MPCLevelCmd()
        self.mpc_level_state = MPCLevelState()
        self.low_level_state = LowLevelState()

        # default command: stand still at the origin
        self.mpc_level_cmd.vel_cmd = np.zeros(3)
        self.mpc_level_cmd.vel_rpy_cmd = np.zeros(3)
        self.mpc_level_cmd.fp_rel_cmd = np.zeros(8)
        self.mpc_level_cmd.fh_rel_cmd = np.zeros(4)
        self.mpc_level_cmd.footswing_height = 0.06
        self.mpc_level_cmd.offsets_smoothed = np.zeros(4).astype(int)
        self.mpc_level_cmd.durations_smoothed = (np.ones(4)*10).astype(int)
        self.mpc_level_cmd.mpc_table_update = np.ones((40, 1)).astype(int)
        self.mpc_level_cmd.vel_table_update = np.zeros((30, 1))
        self.mpc_level_cmd.vel_rpy_table_update = np.zeros((30, 1))
        self.mpc_level_cmd.iterations_table_update = (np.ones(10)*13).astype(int)
        self.mpc_level_cmd.planningHorizon = 10
        self.mpc_level_cmd.adaptationHorizon = 10
        self.mpc_level_cmd.adaptationSteps = 2
        self.mpc_level_cmd.iterationsBetweenMPC = 13

        # wait for first state info before running MPC
        self.robot_state_is_initialized = False

        self.pd_torque_pub = rospy.Publisher("pd_torque_command",
            PDPlusTorqueCommand, queue_size=10)

        self.robot_state_sub = rospy.Subscriber("robot_state",
            RobotState, self.robot_state_callback)

        self.params_sub = rospy.Subscriber("traj_params",
            TrajParams, self.params_callback)

    def robot_state_callback(self, robot_state_msg):
        
        self.mpc_level_state.body_pos = robot_state_msg.body_pos
        self.mpc_level_state.body_rpy = robot_state_msg.body_rpy
        self.mpc_level_state.body_linear_vel = robot_state_msg.body_linear_vel
        self.mpc_level_state.body_angular_vel = robot_state_msg.body_angular_vel
        self.mpc_level_state.body_linear_accel = np.zeros(3) # ??
        self.mpc_level_state.joint_pos = np.array(robot_state_msg.joint_pos)
        self.mpc_level_state.joint_vel = np.array(robot_state_msg.joint_vel)

        self.low_level_state.body_pos = robot_state_msg.body_pos
        self.low_level_state.body_rpy = robot_state_msg.body_rpy
        self.low_level_state.body_linear_vel = robot_state_msg.body_linear_vel
        self.low_level_state.body_angular_vel = robot_state_msg.body_angular_vel
        self.low_level_state.joint_pos = np.array(robot_state_msg.joint_pos)
        self.low_level_state.joint_vel = np.array(robot_state_msg.joint_vel)

        self.robot_state_is_initialized = True

        
    def params_callback(self, traj_params_msg):

        self.mpc_level_cmd.vel_cmd = traj_params_msg.vel_cmd
        self.mpc_level_cmd.vel_rpy_cmd = traj_params_msg.vel_rpy_cmd
        self.mpc_level_cmd.fp_rel_cmd = traj_params_msg.fp_rel_cmd
        self.mpc_level_cmd.fh_rel_cmd = traj_params_msg.fh_rel_cmd
        self.mpc_level_cmd.footswing_height = traj_params_msg.footswing_height
        self.mpc_level_cmd.offsets_smoothed = traj_params_msg.offsets_smoothed
        self.mpc_level_cmd.durations_smoothed = traj_params_msg.durations_smoothed
        self.mpc_level_cmd.mpc_table_update = np.array(traj_params_msg.mpc_table_update).astype(int)
        self.mpc_level_cmd.vel_table_update = traj_params_msg.vel_table_update
        self.mpc_level_cmd.vel_rpy_table_update = traj_params_msg.vel_rpy_table_update
        self.mpc_level_cmd.iterations_table_update = np.array(traj_params_msg.iterations_table_update).astype(int)
        self.mpc_level_cmd.planningHorizon = traj_params_msg.planningHorizon
        self.mpc_level_cmd.adaptationHorizon = traj_params_msg.adaptationHorizon
        self.mpc_level_cmd.adaptationSteps = traj_params_msg.adaptationSteps
        self.mpc_level_cmd.iterationsBetweenMPC = traj_params_msg.iterationsBetweenMPC


    def step_mpc_controller(self):
        if not self.robot_state_is_initialized:
            return

        rot_w_b_cur = inversion(get_rotation_matrix_from_rpy(self.low_level_state.body_rpy))

        self.mpc_controller.nmpc.set_accept_update()
        low_level_cmd = self.mpc_controller.step_with_mpc_table(self.mpc_level_cmd, self.mpc_level_state, self.low_level_state, rot_w_b_cur, override_forces=None)

        pd_torque_cmd = PDPlusTorqueCommand()
        pd_torque_cmd.p_targets = low_level_cmd.p_targets
        pd_torque_cmd.v_targets = low_level_cmd.v_targets
        pd_torque_cmd.p_gains = low_level_cmd.p_gains
        pd_torque_cmd.v_gains = low_level_cmd.v_gains
        pd_torque_cmd.ff_torque = low_level_cmd.ff_torque

        self.pd_torque_pub.publish(pd_torque_cmd)


if __name__ == '__main__':
    rospy.init_node('MPCController', anonymous=True)
    r = rospy.Rate(500)
    mpc_ctrl = MPCController()
    while not rospy.is_shutdown():
        mpc_ctrl.step_mpc_controller()
        r.sleep()
