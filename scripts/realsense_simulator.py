#!/usr/bin/env python

#import imutils
import numpy as np
import rospy

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand, ImageRequest
import sensor_msgs.msg

class SimulatedCamera():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        
        self.camera_request_pub = rospy.Publisher("camera_request",
            ImageRequest, queue_size=10)

        self.camera_pub = rospy.Publisher("camera_image",
            sensor_msgs.msg.Image, queue_size=10)

        self.sim_camera_sub = rospy.Subscriber("simulated_image",
            sensor_msgs.msg.Image, self.image_callback)


    def image_callback(self, camera_msg):
        self.camera_pub.publish(camera_msg)

    def publish_image(self):
        image_request_msg = ImageRequest()
        self.camera_request_pub.publish(image_request_msg)

if __name__ == '__main__':
    rospy.init_node('SimulatedCamera', anonymous=True)
    r = rospy.Rate(10)
    camera = SimulatedCamera()
    while not rospy.is_shutdown():
        camera.publish_image()
        r.sleep()
