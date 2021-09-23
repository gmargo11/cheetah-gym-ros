#!/usr/bin/env python

#import imutils
import numpy as np
import rospy
from cv_bridge import CvBridge

from cheetah_gym_ros.msg import RobotState, PDPlusTorqueCommand, ImageRequest, TerrainHeightmap
import sensor_msgs.msg

from cheetahgym.sensors.heightmap_sensor import HeightmapSensor
from cheetahdeploy.evaluation.evaluation_utils import load_cfg

class MapPublisher():
    """
    A class for asynchronously querying the camera image.
    Subscribes to: /simulated_image (Image) : the depth image from the onboard camera.
    Publishes to: /camera_request (ImageRequest) : Asynchronously queries an image from the simulator.
    Publishes to: /camera_image (Image) : the depth image from the onboard camera.
    """
    def __init__(self):

        self.cfg = load_cfg(rospy.get_param('model_path')).alg

        self.robot_state_is_initialized = False

        self.bridge = CvBridge()

        # initialize heightmap generator from config file
        self.initialize_heightfield_distribution(terrain_cfg_file=self.cfg.terrain_cfg_file)
        self.sample_heightfield()
        
        self.map_pub = rospy.Publisher("terrain_map",
            TerrainHeightmap, queue_size=10)

        self.local_map_pub = rospy.Publisher("local_heightfield",
            TerrainHeightmap, queue_size=10)

        #self.map_request_sub = rospy.Subscriber("local_heightfield_request",
        #    LocalHeightfieldRequest, self.local_heightfield_request_callback)

        self.robot_state_sub = rospy.Subscriber("robot_state",
            RobotState, self.robot_state_callback)



    def robot_state_callback(self, robot_state_msg):
        
        self.robot_body_pos = robot_state_msg.body_pos
        self.robot_body_rpy = robot_state_msg.body_rpy

        self.robot_state_is_initialized = True


    def publish_heightmap(self):
        heightmap = TerrainHeightmap()
        heightmap.heightfield = self.bridge.cv2_to_imgmsg(self.heightmap_sensor.heightmap_array, encoding="passthrough")
        heightmap.n_rows = self.heightmap_sensor.heightmap_array.shape[0]
        heightmap.n_cols = self.heightmap_sensor.heightmap_array.shape[1]
        heightmap.resolution_x = self.heightmap_sensor.hmap_cfg["resolution"]
        heightmap.resolution_y = self.heightmap_sensor.hmap_cfg["resolution"]
        heightmap.resolution_z = 1.

        self.map_pub.publish(heightmap)


    def publish_local_heightfield(self):
        if not self.robot_state_is_initialized:
            return

        local_heightfield = TerrainHeightmap()

        hf = self.heightmap_sensor.get_heightmap_ob(    self.robot_body_pos, 
                                                        self.robot_body_rpy, 
                                                        x_shift=self.cfg.im_x_shift,
                                                        y_shift=self.cfg.im_y_shift,
                                                        im_height=self.cfg.im_height,
                                                        im_width=self.cfg.im_width,
                                                        im_x_resolution=self.cfg.im_x_resolution,
                                                        im_y_resolution=self.cfg.im_y_resolution,
                                                        cfg=None)

        local_heightfield.heightfield = self.bridge.cv2_to_imgmsg(hf, encoding="passthrough")
        local_heightfield.n_rows = hf.shape[0]
        local_heightfield.n_cols = hf.shape[1]
        local_heightfield.resolution_x = self.heightmap_sensor.hmap_cfg["resolution"]
        local_heightfield.resolution_y = self.heightmap_sensor.hmap_cfg["resolution"]
        local_heightfield.resolution_z = 1.

        self.local_map_pub.publish(local_heightfield)


    def initialize_heightfield_distribution(self, terrain_cfg_file):
        self.heightmap_sensor = HeightmapSensor(terrain_cfg_file)


    def sample_heightfield(self):
        self.heightmap_sensor.load_new_heightmap()


if __name__ == '__main__':
    rospy.init_node('MapPublisher', anonymous=True)
    r = rospy.Rate(30)
    map_publisher = MapPublisher()
    while not rospy.is_shutdown():
        map_publisher.publish_heightmap()
        map_publisher.publish_local_heightfield()
        r.sleep()
