#!/usr/bin/env python

#import imutils
import numpy as np
import rospy

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand, ImageRequest, EmergencyStopInfo
import sensor_msgs.msg

class SafetyController():
    """
    A class for ensuring the robot state and joint command are within safety limits.
    Subscribes to: /robot_state (RobotState) : the body and joint state of the robot.
    Subscribes to: /pd_torque_command (PDPlusTorqueCommand) : the commanded pd targets, gains, and feedforward torque.
    Publishes to: /estop_signal (EmergencyStopInfo) : if a safety condition is violated, a message here reports its ID#
    """
    def __init__(self):
        
        self.robot_state_sub = rospy.Subscriber("robot_state",
            RobotState, self.robot_state_callback)

        self.action_sub = rospy.Subscriber("pd_torque_command",
            PDPlusTorqueCommand, self.action_callback)

        self.estop_pub = rospy.Publisher("estop_signal",
            EmergencyStopInfo, queue_size=10)


    def robot_state_callback(self, robot_state_msg):

        MAX_JOINT_SPEED = 30
        MAX_BODY_RP = 0.8
        MIN_BODY_HEIGHT = 0.2

        # robot body too low
        if robot_state_msg.body_pos[2] < MIN_BODY_HEIGHT:
            self.send_ESTOP(condition=1)
        # robot body pitch/yaw too great
        if abs(robot_state_msg.body_rpy[0]) > MAX_BODY_RP or abs(robot_state_msg.body_rpy[1]) > MAX_BODY_RP:
            self.send_ESTOP(condition=2)
        # joints moving too fast
        if np.max(np.abs(np.array(robot_state_msg.joint_vel))) > MAX_JOINT_SPEED:
            self.send_ESTOP(condition=3)

    def action_callback(self, pd_plus_torque_msg):

        MAX_SAFE_TORQUE = 20

        # commanded torque unsafe
        if np.max(np.abs(np.array(pd_plus_torque_msg.ff_torque))) > MAX_SAFE_TORQUE:
            self.send_ESTOP(condition=4)

    def send_ESTOP(self, condition=-1):

        estop_info_msg = EmergencyStopInfo()
        estop_info_msg.condition = condition

        self.estop_pub.publish(estop_info_msg)


if __name__ == '__main__':
    rospy.init_node('SafetyController', anonymous=True)
    r = rospy.Rate(500)
    safety_ctrl = SafetyController()
    rospy.spin()
