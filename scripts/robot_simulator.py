#!/usr/bin/env python

#import imutils
import numpy as np
import rospy
from cv_bridge import CvBridge
import cv2
import time

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand, ImageRequest, EmergencyStopInfo
import sensor_msgs.msg
from rosgraph_msgs.msg import Clock

from cheetahgym.systems.pybullet_system import PyBulletSystem
from cheetahgym.data_types.low_level_types import LowLevelCmd
from cheetahdeploy.evaluation.evaluation_utils import load_cfg
from cheetahgym.data_types.camera_parameters import cameraParameters



class SimulatedRobot():
    """
    A class for simulating the robot dynamics in PyBullet.
    Subscribes to: /pd_torque_command (PDPlusTorqueCommand) : the commanded pd targets, gains, and feedforward torque.
    Subscribes to: /estop_signal (EmergencyStopInfo) : if a safety condition is violated, a message here reports its ID#
    Subscribes to: /camera_request (ImageRequest) : Asynchronously queries an image from the simulator.
    Publishes to: /robot_state (RobotState) : the body and joint state of the robot.
    Publishes to: /simulated_image (Image) : the depth image from the onboard camera.
    """
    def __init__(self, simulator_name="PYBULLET", cfg=None, gui=False, fix_body=False, realtime=False):
        
        self._simulator_name = simulator_name
        self._gui = gui
        self._fix_body = fix_body
        self._realtime = realtime

        self.timestep = 1./500.
        self.num_steps = 0

        self.ESTOP = False

        self.bridge = CvBridge()


        self.gc_init = np.array(
            [0.0, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 0.0, -0.80, 1.6, 0.0, -0.80, 1.6, 0.0, -0.8, 1.6, 0.0,
             -0.8, 1.6, ])

        cfg = load_cfg(rospy.get_param('model_path')).alg

        if self._simulator_name == "PYBULLET":
            cfg.use_egl = True
            #cfg = None
            self.simulator = PyBulletSystem(cfg, gui=self._gui, mpc_controller=None,
                                                initial_coordinates=self.gc_init, fix_body=self._fix_body)

            self.camera_params = cameraParameters(width=cfg.depth_cam_width, height=cfg.depth_cam_height, 
                                x=cfg.depth_cam_x, y=cfg.depth_cam_y, z=cfg.depth_cam_z, 
                                roll=cfg.depth_cam_roll, pitch=cfg.depth_cam_pitch, yaw=cfg.depth_cam_yaw, 
                                fov=cfg.depth_cam_fov, aspect=cfg.depth_cam_aspect, 
                                nearVal=cfg.depth_cam_nearVal, farVal=cfg.depth_cam_farVal,
                                cam_pose_std=cfg.cam_pose_std, cam_rpy_std=cfg.cam_rpy_std)

            self.low_level_cmd = LowLevelCmd()
            # set default low-level command for standing pose
            self.set_standing_targets()

        self.state_pub = rospy.Publisher("robot_state",
            RobotState, queue_size=10)

        self.camera_pub = rospy.Publisher("simulated_image",
            sensor_msgs.msg.Image, queue_size=10)

        self.sim_clock_pub = rospy.Publisher("clock", Clock, queue_size=10)

        self.cam_request_sub = rospy.Subscriber("camera_request",
            ImageRequest, self.camera_callback)

        self.action_sub = rospy.Subscriber("pd_torque_command",
            PDPlusTorqueCommand, self.action_callback)

        self.estop_sub = rospy.Subscriber("estop_signal",
            EmergencyStopInfo, self.emergency_stop_callback)

        # if not self._realtime:
        #     time.sleep(0.1)
        #     self.step_simulation()


    def action_callback(self, pd_plus_torque_msg):

        if self.ESTOP:
            return

        self.low_level_cmd.p_targets = pd_plus_torque_msg.p_targets
        self.low_level_cmd.v_targets = pd_plus_torque_msg.v_targets
        self.low_level_cmd.p_gains = pd_plus_torque_msg.p_gains
        self.low_level_cmd.v_gains = pd_plus_torque_msg.v_gains
        self.low_level_cmd.ff_torque = pd_plus_torque_msg.ff_torque

        #if not self._realtime:
        #    self.step_simulation()

    def camera_callback(self, camera_request_msg):
        depth, rgb = self.simulator.render_camera_image(self.camera_params)

        #cv2_depth = cv2.fromarray(depth)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough")
        self.camera_pub.publish(depth_msg)

    def emergency_stop_callback(self, estop_info_msg):
        self.ESTOP = True
        print(f"ESTOP code {estop_info_msg.condition}")
        self.set_zero_torque()

    def step_simulation(self):
        self.observation = self.simulator.step_state_low_level(self.low_level_cmd, loop_count=1)

        state_msg = RobotState()
        state_msg.body_pos = self.observation.body_pos
        state_msg.body_rpy = self.observation.body_rpy
        state_msg.body_linear_vel = self.observation.body_linear_vel
        state_msg.body_angular_vel = self.observation.body_angular_vel
        state_msg.joint_pos = self.observation.joint_pos
        state_msg.joint_vel = self.observation.joint_vel

        # increment simulation time
        self.num_steps += 1
        sim_clock = Clock()
        sim_clock.clock = rospy.Time.from_sec(self.num_steps * self.timestep)
        self.sim_clock_pub.publish(sim_clock)

        self.state_pub.publish(state_msg)



    def set_standing_targets(self):
        self.low_level_cmd.p_targets = [0.0, -0.80, 1.6, 0.0, -0.80, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6]
        self.low_level_cmd.v_targets = [0.0] * 12
        self.low_level_cmd.p_gains = [40] * 12
        self.low_level_cmd.v_gains = [1] * 12
        self.low_level_cmd.ff_torque = [0] * 12

    def set_zero_torque(self):
        self.low_level_cmd.p_targets = [0.0, -0.80, 1.6, 0.0, -0.80, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6]
        self.low_level_cmd.v_targets = [0.0] * 12
        self.low_level_cmd.p_gains = [0] * 12
        self.low_level_cmd.v_gains = [0] * 12
        self.low_level_cmd.ff_torque = [0] * 12


if __name__ == '__main__':
    rospy.init_node('SimulatedRobot', anonymous=True)
    #r = rospy.Rate(500)
    robot = SimulatedRobot(gui=True, realtime=False)

    while not rospy.is_shutdown():
        robot.step_simulation()
        #r.sleep()
        #time.sleep(0.01)

