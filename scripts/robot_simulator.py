#!/usr/bin/env python

#import imutils
import numpy as np
import rospy

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand

from cheetahgym.systems.pybullet_system import PyBulletSystem
from cheetahgym.data_types.low_level_types import LowLevelCmd
from cheetahdeploy.evaluation.evaluation_utils import load_cfg

class SimulatedRobot():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self, simulator_name="PYBULLET", gui=False, fix_body=False):
        
        self._simulator_name = simulator_name
        self._gui = gui
        self._fix_body = fix_body


        self.gc_init = np.array(
            [0.0, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 0.0, -0.80, 1.6, 0.0, -0.80, 1.6, 0.0, -0.8, 1.6, 0.0,
             -0.8, 1.6, ])

        if self._simulator_name == "PYBULLET":
            cfg = load_cfg() # TODO: enable specification of config file!
            #cfg = None
            self.simulator = PyBulletSystem(cfg.alg, gui=self._gui, mpc_controller=None,
                                                initial_coordinates=self.gc_init, fix_body=self._fix_body)
            self.low_level_cmd = LowLevelCmd()
            # set default low-level command for standing pose
            self.low_level_cmd.p_targets = [0.0, -0.80, 1.6, 0.0, -0.80, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6]
            self.low_level_cmd.v_targets = [0.0] * 12
            self.low_level_cmd.p_gains = [40] * 12
            self.low_level_cmd.v_gains = [1] * 12
            self.low_level_cmd.ff_torque = [0] * 12

        self.state_pub = rospy.Publisher("robot_state",
            RobotState, queue_size=10)

        self.action_sub = rospy.Subscriber("pd_torque_command",
            PDPlusTorqueCommand, self.action_callback)



    def action_callback(self, pd_plus_torque_msg):

        self.low_level_cmd.p_targets = pd_plus_torque_msg.p_targets
        self.low_level_cmd.v_targets = pd_plus_torque_msg.v_targets
        self.low_level_cmd.p_gains = pd_plus_torque_msg.p_gains
        self.low_level_cmd.v_gains = pd_plus_torque_msg.v_gains
        self.low_level_cmd.ff_torque = pd_plus_torque_msg.ff_torque
        

    def step_simulation(self):
        self.observation = self.simulator.step_state_low_level(self.low_level_cmd, loop_count=1)

        state_msg = RobotState()
        state_msg.body_pos = self.observation.body_pos
        state_msg.body_rpy = self.observation.body_rpy
        state_msg.body_linear_vel = self.observation.body_linear_vel
        state_msg.body_angular_vel = self.observation.body_angular_vel
        state_msg.joint_pos = self.observation.joint_pos
        state_msg.joint_vel = self.observation.joint_vel

        self.state_pub.publish(state_msg)


if __name__ == '__main__':
    rospy.init_node('SimulatedRobot', anonymous=True)
    r = rospy.Rate(500)
    robot = SimulatedRobot(gui=True)
    while not rospy.is_shutdown():
        robot.step_simulation()
        r.sleep()
