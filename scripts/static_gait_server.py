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


def get_mpc_table(offsets, durations, gait_cycle_len=10):
        mpc_table = np.zeros((4, gait_cycle_len))
        for i in range(len(offsets)):
            start = int(offsets[i])
            end = int(offsets[i] + durations[i])
            if end < gait_cycle_len:
                    mpc_table[i, start:end] = 1
            else:
                mpc_table[i, start:] = 1
                mpc_table[i, :(end % gait_cycle_len)] = 1
        return mpc_table


class StaticGaitServer():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self, iterationsBetweenMPC=17):

        self.robot_state_is_initialized = False
        self.iterationsBetweenMPC = iterationsBetweenMPC
        self.iteration_counter = 0

        self.robot_state_sub = rospy.Subscriber("robot_state",
            RobotState, self.robot_state_callback)

        self.traj_params_pub = rospy.Publisher("traj_params",
            TrajParams, queue_size=10)

    def robot_state_callback(self, robot_state_msg):
       
        self.robot_state_is_initialized = True


    def publish_static_gait(self, GAIT_TYPE="STANDING"):
        if not self.robot_state_is_initialized:
            return

        self.iteration_counter += 1

        traj_params_msg = TrajParams()
        traj_params_msg.vel_cmd = [0, 0, 0]
        traj_params_msg.vel_rpy_cmd = [0, 0, 0]
        traj_params_msg.fp_rel_cmd = [0] * 8
        traj_params_msg.fh_rel_cmd = [0] * 4
        traj_params_msg.footswing_height = 0.06
        traj_params_msg.vel_table_update = [0] * 30
        traj_params_msg.vel_rpy_table_update = [0] * 30
        traj_params_msg.iterations_table_update = [int(self.iterationsBetweenMPC)] * 10
        traj_params_msg.planningHorizon = 10
        traj_params_msg.adaptationHorizon = 10
        traj_params_msg.adaptationSteps = 2
        traj_params_msg.iterationsBetweenMPC = int(self.iterationsBetweenMPC)


        if GAIT_TYPE == "STANDING":
            traj_params_msg.offsets_smoothed = [0] * 4
            traj_params_msg.durations_smoothed = [10] * 4
            traj_params_msg.mpc_table_update = [1] * 40
        elif GAIT_TYPE == "TROTTING":
            traj_params_msg.offsets_smoothed = [self.iteration_counter % 10, (self.iteration_counter+5)%10, (self.iteration_counter+5)%10, self.iteration_counter % 10]
            traj_params_msg.durations_smoothed = [5] * 4
            traj_params_msg.mpc_table_update = get_mpc_table(traj_params_msg.offsets_smoothed, traj_params_msg.durations_smoothed).flatten().astype(int).tolist()
        elif GAIT_TYPE == "PRONKING":
            traj_params_msg.offsets_smoothed = [self.iteration_counter % 10, self.iteration_counter % 10, self.iteration_counter % 10, self.iteration_counter % 10]
            traj_params_msg.durations_smoothed = [5] * 4
            traj_params_msg.mpc_table_update = get_mpc_table(traj_params_msg.offsets_smoothed, traj_params_msg.durations_smoothed).flatten().astype(int).tolist()

        self.traj_params_pub.publish(traj_params_msg)



        

if __name__ == '__main__':
    rospy.init_node('StaticGaitServer', anonymous=True)
    iterationsBetweenMPC = 17.
    r = rospy.Rate(500./iterationsBetweenMPC)
    static_gait_server = StaticGaitServer(iterationsBetweenMPC=iterationsBetweenMPC)
    while not rospy.is_shutdown():
        static_gait_server.publish_static_gait(GAIT_TYPE="PRONKING")
        r.sleep()
