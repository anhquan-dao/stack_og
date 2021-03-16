#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

import rospy
import tf as ros_tf
import tf2_ros
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3


class OccupancyGridCallback():
    def __init__(self, baselink_frame, output_shape, output_resolution, input_topic):
        self.sensor_frame = str()
        self.width = 0
        self.height = 0
        self.resolution = 0
        self.last_time = rospy.Time()

        self.baselink = baselink_frame
        self.output_shape = output_shape
        self.output_resolution = output_resolution

        self.data_mat = np.zeros(self.output_shape, np.uint8)
        self.baselink_frame_mat = np.zeros(self.output_shape, np.uint8)

        self.sensor_frame_crop_tf = [[0, 0, 0], [0, 0, 0, 1]]
        self.crop_mat = np.ndarray
        self.resized_mat = np.zeros_like(self.data_mat)
        self.tf_to_baselink_mat = np.eye(3)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self.input_topic = input_topic
        self.og_sub = rospy.Subscriber(
            self.input_topic, OccupancyGrid, self.callback, queue_size=1)

        self.last_update_time = 0

        time.sleep(1)

    def callback(self, msg):
        self.last_update_time = rospy.Time.now().to_sec()
        self.last_time = msg.header.stamp

        self.sensor_frame = msg.header.frame_id
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution

        self.data_mat = np.array(msg.data, np.uint8).reshape(
            self.height, self.width)
        # print(self.data_mat.dtype)
        self.data_mat += 1

        # Something wrong with this section yielding null shape in resized_mat
        # Please check
        resized_ratio = self.output_resolution/self.resolution
        self.resized_mat = cv2.resize(
            self.data_mat, (0, 0), fx=resized_ratio, fy=resized_ratio)

        self.sensor_frame_crop_tf[0] = [msg.info.origin.position.x,
                                        msg.info.origin.position.y,
                                        msg.info.origin.position.z]

        self.sensor_frame_crop_tf[1] = [msg.info.origin.orientation.x,
                                        msg.info.origin.orientation.y,
                                        msg.info.origin.orientation.z,
                                        msg.info.origin.orientation.w]

    def tf_to_baselink_frame(self):
        '''
        Assuming that the sensor data is centered around the origin of the sensor frame
        Transform sensor data to base_link frame
        Crop the sensor data with pose information
        '''
        if self.last_update_time == 0:
            rospy.logerr_throttle(
                1, "Topic " + self.input_topic + " has not been published")
            return
        elif abs(self.last_update_time - rospy.Time.now().to_sec()) > 1:
            rospy.logwarn_throttle(
                1, "Topic " + self.input_topic + " has not been updated for more than 1 second")
            return

        uncrop_tf_to_baselink_mat = self.get_tf_to_baselink_mat(
            self.resized_mat, self.sensor_frame, self.baselink)
        crop_mat = self.get_sensor_frame_crop_mat(self.sensor_frame_crop_tf)

        self.tf_to_baselink_mat = np.matmul(
            uncrop_tf_to_baselink_mat, crop_mat)
        # print(self.tf_to_baselink_mat)
        self.baselink_frame_mat = cv2.warpAffine(
            self.resized_mat, self.tf_to_baselink_mat[:2, :], self.output_shape)
        # print(self.crop_mat.dtype)

        # print("Done transforming sensor data to baselink frame")

    def get_sensor_frame_crop_mat(self, crop_tf):
        # Assume that we are doing 2D OccupancyGrid only
        theta_z = np.arctan(crop_tf[1][2]/crop_tf[1][3])
        twod_quaternion = [0, 0, np.sin(theta_z), np.cos(theta_z)]

        og_rot_mat = ros_tf.transformations.quaternion_matrix(twod_quaternion)[
            :3, :3]

        warp_affine_mat = og_rot_mat
        warp_affine_mat[0, 2] = self.output_shape[0] / \
            2 + crop_tf[0][0]/self.output_resolution
        warp_affine_mat[1, 2] = self.output_shape[1] / \
            2 + crop_tf[0][1]/self.output_resolution

        return warp_affine_mat

    def get_tf_to_baselink_mat(self, og_np, sensor_frame, baselink_frame):
        tf_to_baselink_trans = Vector3()

        tf_to_baselink_rot = Quaternion()
        tf_to_baselink_rot.w = 1

        if not sensor_frame == "":
            tf_to_baselink = self.tfBuffer.lookup_transform(
                sensor_frame, baselink_frame, rospy.Time(0))

            tf_to_baselink_trans, tf_to_baselink_rot = tf_to_baselink.transform.translation, tf_to_baselink.transform.rotation
        else:
            rospy.logerr_throttle(1, "Sensor frame for topic <" +
                                  self.input_topic + "> is empty, possibly topic has not been published")
        '''
		try:
			tf_to_baselink_trans, tf_to_baselink_rot = self.tf_listener.lookupTransform(self.sensor_frame, baselink_frame, 
																						rospy.Time(0))
		except (ros_tf.ExtrapolationException, ros_tf.LookupException) as e:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(e).__name__, e.args)
			rospy.logerr(message)
			continue
		'''

        theta_z = np.arctan(tf_to_baselink_rot.z/tf_to_baselink_rot.w) * 2

        og_center = tuple(np.array(og_np.shape[1::-1]) / 2)
        og_rot_mat = np.identity(3)
        og_rot_mat[0:2, :] = cv2.getRotationMatrix2D(
            og_center, np.degrees(theta_z), 1.0)

        og_trans_mat = np.identity(3)
        og_trans_mat[0:2, 2] = np.array(
            [-tf_to_baselink_trans.x, -tf_to_baselink_trans.y])/self.output_resolution

        warp_affine_mat = np.matmul(og_rot_mat, og_trans_mat)

        return warp_affine_mat


if __name__ == "__main__":
    rospy.init_node("occ_grid_handler_test")

    test = OccupancyGridCallback("/base_link", (400, 400), 0.1, '/map/abc')
    # rospy.Subscriber('/map/abc', OccupancyGrid, test.callback), queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
