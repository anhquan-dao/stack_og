#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

import rospy
import tf as ros_tf
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

class OccupancyGridCallback():
    def __init__(self, baselink_frame, output_shape, output_resolution):
        self.width = 0
        self.height = 0
        self.resolution = 0

        self.baselink = baselink_frame
        self.output_shape = output_shape
        self.output_resolution = output_resolution;

        self.data_mat = np.ndarray()
        self.baselink_frame_mat = np.zeros(self.output_shape, np.uint8)

        self.sensor_frame_crop_tf = ([],[])
        self.crop_mat = np.ndarray()

        self.tf_listener = ros_tf.TransformListener()


    def callback(self, msg):
        
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution

        self.data_mat = np.array(msg.data).reshape(self.height, self.width)
        self.data_mat = (self.data_mat+1).round().astype(np.uint8)

        resized_ratio = self.output_resolution/self.resolution
        self.resized_mat = cv2.resize(self.data_mat, (0,0), fx=resized_ratio, fy=resized_ratio)

        self.sensor_frame_crop_tf[0] = [msg.info.origin.position.x,
									    msg.info.origin.position.y,
									    msg.info.origin.position.z]

        self.sensor_frame_crop_tf[1] = = [msg.info.origin.orientation.x,
									      msg.info.origin.orientation.y,
									      msg.info.origin.orientation.z,
									      msg.info.origin.orientation.w]

        self.crop_mat = self.sensor_frame_crop(self.resized_mat, self.sensor_frame_crop_tf)
        
        self.baselink_frame_mat = self.tf_to_baselink_frame(self.crop_mat, self.baselink)

    def sensor_frame_crop(self, og_np, crop_tf):
		# Assume that we are doing 2D OccupancyGrid only
		theta_z             = np.arctan(crop_tf[1][2]/crop_tf[1][3])
		twod_quaternion     = [0, 0, np.sin(theta_z), np.cos(theta_z)]

		og_rot_mat          = ros_tf.transformations.quaternion_matrix(twod_quaternion)
		og_rot_mat          = og_rot_mat[0:2, 0:3]

		og_trans_mat        = np.zeros(2)
		og_trans_mat[0]     = self.output_shape[0]/2  + crop_tf[0][0]/self.output_resolution
		og_trans_mat[1]     = self.output_shape[1]/2  + crop_tf[0][1]/self.output_resolution

		warp_affine_mat       = og_rot_mat
		warp_affine_mat[:, 2] = og_trans_mat

		og_sensor_frame       = cv2.warpAffine(og_np, warp_affine_mat, self.output_shape)

		return og_sensor_frame

	def tf_to_baselink_frame(self, og_np, sensor_frame):
		tf_to_baselink_trans, tf_to_baselink_rot = self.tf_listener.lookupTransform(sensor_frame, 
                                                                                    self.base_link, 
                                                                                    self.tf_listener.getLatestCommonTime(sensor_frame, self.baselink)
		
        theta_z    = np.arctan(tf_to_baselink_rot[2]/tf_to_baselink_rot[3]) * 2

		og_center  = tuple(np.array(og_np.shape[1::-1]) / 2)
		og_rot_mat = np.identity(3)
		og_rot_mat[0:2, :]    = cv2.getRotationMatrix2D(og_center, np.degrees(theta_z), 1.0)

		og_trans_mat   		  = np.identity(3)
		og_trans_mat[0:2,2]   = np.array([-tf_to_baselink_trans[0], -tf_to_baselink_trans[1]])/self.output_resolution

		warp_affine_mat       = np.matmul(og_rot_mat, og_trans_mat)

		warp_affine_mat       = warp_affine_mat[0:2, :]
		og_baselink_frame     = cv2.warpAffine(og_np, warp_affine_mat, self.output_shape)

		return og_baselink_frame

if __name__ == "__main__":
    rospy.init_node("occ_grid_handler_test")

    test = OccupancyGridCallback("/base_link", (40,40), 0.1)
    rospy.Subscriber('/map/free_local_occupancy_grid', OccupancyGrid, test.callback, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
