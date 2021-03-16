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
from geometry_msgs.msg import TransformStamped

x_distance = 0
y_distance = 0
class OccupancyGridStack():
	def __init__(self, fixed_frame = "map", base_link = "base_link", 
					   resolution = 0.1, output_size = [40, 40], 
					   topics_list = list()):

		self.fixed_frame = fixed_frame
		self.base_link = base_link
		self.resolution = resolution 
		self.output_size = output_size
		self.output_shape = tuple([int(round(i/self.resolution)) for i in self.output_size])

		self.current_stack = np.zeros(self.output_shape, np.uint8)
		self.current_stack_int = np.ndarray(self.output_shape, np.int8)
		self.current_stack_float = np.zeros(self.output_shape, np.float16)
		self.temp_stack = np.zeros(self.output_shape, np.uint8)

		self.last_to_current_warp_affine_mat = np.ndarray

		self.new_ROI = np.zeros(self.output_shape, np.uint8)
		self.decay_mask = np.zeros(self.output_shape, np.float16)

		self.map_msg = OccupancyGrid()
		self.og_publisher = rospy.Publisher("test/output", OccupancyGrid, queue_size=1)

		self.tfBuffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
		self.tf2_ros_exception = (tf2_ros.LookupException, tf2_ros.ExtrapolationException)

		self.last_current_pose = TransformStamped()
		self.last_current_pose.transform.rotation.w = 1
		while True:
			try:
				self.last_current_pose = self.tfBuffer.lookup_transform(self.base_link,
																self.fixed_frame,
																rospy.Time.now(),
																rospy.Duration(0.1))
			except self.tf2_ros_exception as e:
				print(e)
				continue
			
			break
		

		self.input_sub = list()
		for i in topics_list:
			self.input_sub.append(rospy.Subscriber(i, OccupancyGrid, self.callback, queue_size = 1))

	def tf_stack_to_current_pose(self):
		global tf2_ros_exception
		global x_distance
		global y_distance
		#----------------------------------------------------------------------------------------
		# Calculate last and current pose transformation in FIXED frame

		last_translation, last_rotation = self.last_current_pose.transform.translation, self.last_current_pose.transform.rotation
		
		last_theta_z		     = np.arctan(last_rotation.z/last_rotation.w) * 2

		last_warp_affine_mat = np.identity(3)
		last_warp_affine_mat[0,0] = np.cos(last_theta_z)
		last_warp_affine_mat[1,1] = last_warp_affine_mat[0,0]
		last_warp_affine_mat[0,1] = -np.sin(last_theta_z)
		last_warp_affine_mat[1,0] = -last_warp_affine_mat[0,1]
		last_warp_affine_mat[0,2] = last_translation.x/self.resolution
		last_warp_affine_mat[1,2] = last_translation.y/self.resolution

		#----------------------------------------------------------------------------------------
		current_translation, current_rotation  = last_translation, last_rotation
		
		# Assuming that the transform from last pose to current pose remains the same in a small time window
		# if no transform available, reuse the last transform
		try:
			current_tf = self.tfBuffer.lookup_transform(self.base_link, self.fixed_frame, rospy.Time.now(), rospy.Duration(0.1))
			current_translation, current_rotation  = current_tf.transform.translation, current_tf.transform.rotation

			current_theta_z		        = np.arctan(current_rotation.z/current_rotation.w) * 2
			
			current_warp_affine_mat = np.identity(3)
			current_warp_affine_mat[0,0] = np.cos(current_theta_z)
			current_warp_affine_mat[1,1] = current_warp_affine_mat[0,0]
			current_warp_affine_mat[0,1] = -np.sin(current_theta_z)
			current_warp_affine_mat[1,0] = -current_warp_affine_mat[0,1]
			current_warp_affine_mat[0,2] = current_translation.x/self.resolution
			current_warp_affine_mat[1,2] = current_translation.y/self.resolution
			
			#---------------------------------------------------------------------------last_to_current_warp_affine_mat[1,3] + self.output_shape[1]/2-------------
			last_to_current_warp_affine_mat = np.matmul(current_warp_affine_mat,np.linalg.inv(last_warp_affine_mat))

			x_distance += last_to_current_warp_affine_mat[0,2]
			y_distance += last_to_current_warp_affine_mat[1,2]

			translate_to_center = np.identity(3)
			translate_to_center[0,2] = -self.output_shape[0] / 2
			translate_to_center[1,2] = -self.output_shape[1] / 2

			last_to_current_warp_affine_mat = np.matmul(np.linalg.inv(translate_to_center), last_to_current_warp_affine_mat)
			last_to_current_warp_affine_mat = np.matmul(last_to_current_warp_affine_mat, translate_to_center)
			
			self.last_to_current_warp_affine_mat = last_to_current_warp_affine_mat[:2, :]
			
			x_distance += self.last_to_current_warp_affine_mat[0,2]
			y_distance += self.last_to_current_warp_affine_mat[1,2]
			print(x_distance)
			print(y_distance)

			#print(self.last_to_current_warp_affine_mat)
			print("---------")
		except self.tf2_ros_exception as e:
			template = "{0} occurred at transforming stack to current pose.\n Arguments:\n{1!r}"
			message = template.format(type(e).__name__, e.args)
			rospy.logerr(message)

		self.current_stack = cv2.warpAffine(self.current_stack, self.last_to_current_warp_affine_mat, 
												 self.output_shape)
											 
		self.last_current_pose.transform.translation, self.last_current_pose.transform.rotation = (current_translation, current_rotation)

	def publish_og_msg(self):

		self.BATOLOPU(self.temp_stack)
		self.temp_stack = np.zeros(self.output_shape, np.uint8)
		
		self.current_stack_int = self.current_stack.round().astype(np.int8) - 1

		self.map_msg.header.frame_id = self.base_link
		self.map_msg.header.stamp    = rospy.Time.now()

		self.map_msg.info.height = self.output_shape[0]      #Unit: Pixel
		self.map_msg.info.width  = self.output_shape[1]      #Unit: Pixel
		self.map_msg.info.resolution = self.resolution

		self.map_msg.info.origin.position.x = -self.output_size[0]/2      #Unit: Meter
		self.map_msg.info.origin.position.y = -self.output_size[1]/2      #Unit: Meter
		self.map_msg.info.origin.position.z = 0
		self.map_msg.info.origin.orientation.x = 0
		self.map_msg.info.origin.orientation.y = 0
		self.map_msg.info.origin.orientation.z = 0
		self.map_msg.info.origin.orientation.w = 1

		self.map_msg.data =  self.current_stack_int.flatten().tolist()
		self.map_msg.info.map_load_time = rospy.Time.now()

		self.og_publisher.publish(self.map_msg)

	def callback(self, msg):
		sensor_frame = msg.header.frame_id
		width = msg.info.width
		height = msg.info.height
		input_resolution = msg.info.resolution

		data_mat = np.array(msg.data).reshape(height, width)
		data_mat = (data_mat+1).round().astype(np.uint8)

		resized_ratio = self.resolution/input_resolution
		resized_mat = cv2.resize(data_mat, (0,0), fx=resized_ratio, fy=resized_ratio)


		sensor_frame_crop_tf = [[],[]]
		sensor_frame_crop_tf[0] = [msg.info.origin.position.x,
										msg.info.origin.position.y,
										msg.info.origin.position.z]

		sensor_frame_crop_tf[1] = [msg.info.origin.orientation.x,
										msg.info.origin.orientation.y,
										msg.info.origin.orientation.z,
										msg.info.origin.orientation.w]

		crop_mat = self.sensor_frame_crop(resized_mat, sensor_frame_crop_tf)
		#print(crop_mat.dtype)
		baselink_frame_mat = self.tf_to_baselink_frame(crop_mat, sensor_frame, self.base_link)
		#print(baselink_frame_mat.dtype)
	
		tf_to_baselink= self.tfBuffer.lookup_transform(sensor_frame,
													   self.base_link,
													   rospy.Time.now(), rospy.Duration(0.1))
		
		self.tf_stack_to_current_pose()
		self.temp_stack = np.maximum(self.temp_stack, baselink_frame_mat)
		

	def tf_to_baselink_frame(self, og_np, sensor_frame, baselink_frame):
		tf_to_baselink_trans = [0,0,0]
		tf_to_baselink_rot   = [0,0,0,1]

		try:
			tf_to_baselink= self.tfBuffer.lookup_transform(sensor_frame,
															baselink_frame,
															rospy.Time.now(), rospy.Duration(0.1))
			
			tf_to_baselink_trans, tf_to_baselink_rot = tf_to_baselink.transform.translation, tf_to_baselink.transform.rotation
			theta_z    = np.arctan(tf_to_baselink_rot.z/tf_to_baselink_rot.w) * 2

			og_center  = tuple(np.array(og_np.shape[1::-1]) / 2)
			og_rot_mat = np.identity(3)
			og_rot_mat[0:2, :]    = cv2.getRotationMatrix2D(og_center, np.degrees(theta_z), 1.0)

			og_trans_mat   		  = np.identity(3)
			og_trans_mat[0:2,2]   = np.array([-tf_to_baselink_trans.x, -tf_to_baselink_trans.y])/self.resolution

			warp_affine_mat       = np.matmul(og_rot_mat, og_trans_mat)

			warp_affine_mat       = warp_affine_mat[0:2, :]
			
			return cv2.warpAffine(og_np, warp_affine_mat, self.output_shape)

		except self.tf2_ros_exception as e:
			# If no transform available, discard the sensor data, return zero matrix
			template = "{0} occurred at transforming sensor data to baselink frame.\n Arguments:\n{1!r}"
			message = template.format(type(e).__name__, e.args)
			rospy.logerr(message)

			return np.zeros(self.output_shape, np.int8)

		

	def sensor_frame_crop(self, og_np, crop_tf):
		# Assume that we are doing 2D OccupancyGrid only
		theta_z             = np.arctan(crop_tf[1][2]/crop_tf[1][3])
		twod_quaternion     = [0, 0, np.sin(theta_z), np.cos(theta_z)]

		og_rot_mat          = ros_tf.transformations.quaternion_matrix(twod_quaternion)
		og_rot_mat          = og_rot_mat[0:2, 0:3]

		og_trans_mat        = np.zeros(2)
		og_trans_mat[0]     = self.output_shape[0]/2  + crop_tf[0][0]/self.resolution
		og_trans_mat[1]     = self.output_shape[1]/2  + crop_tf[0][1]/self.resolution

		warp_affine_mat       = og_rot_mat
		warp_affine_mat[:, 2] = og_trans_mat

		return cv2.warpAffine(og_np, warp_affine_mat, self.output_shape)

	def BATOLOPU(self, input_occ_grid):
		#stacked = (og_current_pose + baselink_transformed_og*4)/5
		# batolopu: BAselink Transformed Occupancygrid LOw Priotizing Unknown
		# If new OccupancyGrid has unknown data, then unknown data is filled by 
		# stacked OccupancyGrid
		
		'''
		batolopu = (input_occ_grid + (input_occ_grid == 0) * self.current_stack) + (input_occ_grid == 1) * input_occ_grid
		
		self.current_stack = (self.current_stack + batolopu) / 2
		'''

		# Adaptable weights
		a_ = 0
		b_ = 1 - a_
		batolopu = ((input_occ_grid + (input_occ_grid == 0) * self.current_stack_float) 
		            + (input_occ_grid == 1) * input_occ_grid).astype(np.float16)

		self.current_stack_float = (self.current_stack.astype(np.float16) * a_ + batolopu * b_)
		self.current_stack = (self.current_stack_float).astype(np.uint8)	


if __name__ == "__main__":
	rospy.init_node("stack_og")

	fixed_frame = rospy.get_param("fixed_frame", "odom")
	baselink_frame = rospy.get_param("baselink_frame", "base_link")

	resolution = rospy.get_param("resolution", 0.1)
	height = rospy.get_param("height", 40)
	width = rospy.get_param("width", 40)

	topic_list = rospy.get_param("topic_list", ["/map/image_segmentation", "output"])

	stack_og = OccupancyGridStack(fixed_frame, baselink_frame,
								  resolution, [height, width], topic_list)

	rate = rospy.Rate(100)
	while not rospy.is_shutdown():
		rospy.loginfo("New cycle")
		t1 = time.time()
		stack_og.publish_og_msg()
		t2 = time.time()
		rospy.loginfo(t2 - t1)
		rate.sleep()


	


	

		