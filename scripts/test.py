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

from occ_grid_handler import OccupancyGridCallback

def cv2_og_visualize(name, og_np):
	show = og_np.astype(np.uint8)
	show[show == 1] = 255
	cv2.imshow(name, cv2.flip(show,0))
	cv2.waitKey(1)

class OccupancyGridStack():
	def __init__(self, fixed_frame = "map", base_link = "base_link", 
					   resolution = 0.1, output_size = [40, 40], 
					   topics_list = list(), threshold_value = 50,
					   discard_time = 120):

		self.fixed_frame = fixed_frame
		self.base_link = base_link
		self.resolution = resolution 
		self.output_size = output_size
		self.output_shape = tuple([int(round(i/self.resolution)) for i in self.output_size])

		self.current_stack_unsigned = np.zeros(self.output_shape, np.uint8)
		self.current_stack_signed = np.zeros(self.output_shape, np.int8)
		self.current_stack_float = np.zeros(self.output_shape, np.float16)


		self.last_to_current_warp_affine_mat = np.ndarray((2,3))

		# Using exponential decay to discard information that has not been
		# update but still in the current_stack zone
		self.discard_time = discard_time
		self.decay_rate = np.log(0.01) / discard_time
		self.decay_mask = np.ones(self.output_shape, np.float32) * 100 
		self.last_decay = time.time()


		self.temp_stack = np.zeros(self.output_shape, np.uint16)
		self.threshold_value = threshold_value

		self.map_msg = OccupancyGrid()
		self.og_publisher = rospy.Publisher("test/output", OccupancyGrid, queue_size=1)

		self.tfBuffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
		self.tf2_ros_exception = (tf2_ros.LookupException, tf2_ros.ExtrapolationException)

		self.last_current_pose = TransformStamped()
		self.last_current_pose.transform.rotation.w = 1

		# Wait for tfBuffer to work properly
		wait_time = time.time()
		print(self.base_link)
		print(self.fixed_frame)
		while True:
			if time.time() - wait_time > 2:
				rospy.logerr("Cannot get information about tf. Abort")
				rospy.signal_shutdown("Cannot get information about tf. Abort")
				break

			try:
				self.last_current_pose = self.tfBuffer.lookup_transform(self.base_link,
																self.fixed_frame,
																rospy.Time(0))
			except self.tf2_ros_exception as e:
				template = "An exception of type {0} occurred at __init__. Arguments:\n{1!r}"
				message = template.format(type(e).__name__, e.args)
				rospy.logwarn_throttle(1, message)
				continue

			except tf2_ros.ConnectivityException:
				template = "An exception of type {0} occurred at __init__. Arguments:\n{1!r}"
				message = template.format(type(e).__name__, e.args)
				rospy.logwarn_throttle(1, message)
				continue
			
			break
		

		self.og_handler_holder = list()
		for topic in topics_list:
			self.og_handler_holder.append(OccupancyGridCallback(self.base_link, self.output_shape, self.resolution, topic))

	def tf_stack_to_current_pose(self):
		#start_time = time.time()
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
		#try:
		current_tf = self.tfBuffer.lookup_transform(self.base_link, self.fixed_frame, rospy.Time(0))
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

		translate_to_center = np.identity(3)
		translate_to_center[0,2] = -self.output_shape[0] / 2
		translate_to_center[1,2] = -self.output_shape[1] / 2

		last_to_current_warp_affine_mat = np.matmul(np.linalg.inv(translate_to_center), last_to_current_warp_affine_mat)
		last_to_current_warp_affine_mat = np.matmul(last_to_current_warp_affine_mat, translate_to_center)
		
		self.last_to_current_warp_affine_mat = last_to_current_warp_affine_mat[:2, :]

		#print(self.last_to_current_warp_affine_mat)
		#print("---------")
		#except self.tf2_ros_exception as e:
		#	template = "{0} occurred at transforming stack to current pose.\n Arguments:\n{1!r}"
		#	message = template.format(type(e).__name__, e.args)
		#	rospy.logerr(message)

		self.current_stack_unsigned = cv2.warpAffine(self.current_stack_unsigned, self.last_to_current_warp_affine_mat, 
											self.output_shape)
		self.decay_mask = cv2.warpAffine(self.decay_mask, self.last_to_current_warp_affine_mat, 
										 self.output_shape, cv2.INTER_NEAREST, borderMode = 0, borderValue = 100)						 
		self.last_current_pose.transform.translation, self.last_current_pose.transform.rotation = (current_translation, current_rotation)

		#end_time = time.time()
		#rospy.loginfo("tf_stack_to_current_pose rate: " + str(1/(end_time-start_time)))

	def publish_og_msg(self):

		self.tf_stack_to_current_pose()
		self.temp_stack *= 0
		for og_handler in self.og_handler_holder:
			og_handler.tf_to_baselink_frame()
			self.temp_stack = np.maximum(self.temp_stack, og_handler.baselink_frame_mat)
			#print(og_handler.baselink_frame_mat.dtype)

		#cv2_og_visualize("temp stack", self.temp_stack)
		
		#print(self.temp_stack.dtype)
		self.BATOLOPU(self.temp_stack)
		#cv2_og_visualize("current stack", self.current_stack_signed)
			
		#self.current_stack_signed = self.current_stack_unsigned.round().astype(np.int8) - 1

		occupancy_grid = self.current_stack_signed.flatten()

		self.map_msg.header.frame_id = self.base_link
		self.map_msg.header.stamp    = rospy.Time.now()

		self.map_msg.info.height = self.output_shape[0]      #Unit: Pixel
		self.map_msg.info.width  = self.output_shape[1]      #Unit: Pixel
		self.map_msg.info.resolution = self.resolution

		self.map_msg.info.origin.position.x = -self.output_size[0]/2      #Unit: Meter
		self.map_msg.info.origin.position.y = -self.output_size[1]/2      #Unit: Meter
		self.map_msg.info.origin.position.z = 0

		self.map_msg.data = occupancy_grid.tolist()
		self.map_msg.info.map_load_time = rospy.Time.now()

		self.og_publisher.publish(self.map_msg)		

	def BATOLOPU(self, input_occ_grid):
		#stacked = (og_current_pose + baselink_transformed_og*4)/5
		# batolopu: BAselink Transformed Occupancygrid LOw Priotizing Unknown
		# If new OccupancyGrid has unknown data, then unknown data is filled by 
		# stacked OccupancyGrid
		
		start_time = time.time()		
		
		if time.time() - self.last_decay > 1:
			start_time = time.time()
			self.decay_mask[(self.current_stack_unsigned > 0)] *= np.exp(-self.decay_rate)
			self.decay_mask[input_occ_grid > 0] = 100
			self.decay_mask[self.decay_mask <= 2] = 0

			self.last_decay = time.time()
			#rospy.loginfo("decay process rate: " + str(1/(self.last_decay-start_time)))

		cv2.imshow("Decay mask", cv2.flip(self.decay_mask.astype(np.uint8), 0))
		cv2.waitKey(1)
		
		batolopu = (input_occ_grid + (input_occ_grid == 0) * self.current_stack_unsigned)

		self.current_stack_float = (batolopu * 0.7 + self.current_stack_unsigned * 0.3)
		self.current_stack_unsigned = self.current_stack_float.round().astype(np.uint8)

		# self.current_stack_unsigned *= (self.decay_mask > 0)

		# self.current_stack_signed = (self.current_stack_unsigned >= self.threshold_value) * 100
		self.current_stack_signed = cv2.threshold(self.current_stack_unsigned, self.threshold_value, 100, cv2.THRESH_BINARY)[1].astype(np.int8)
		self.current_stack_signed[self.current_stack_unsigned == 0] = -1
		
		'''
		kernel = np.ones((5, 5), np.uint8) 
		dilate_kernel = np.ones((15,15), np.uint8)

		mask = (self.current_stack_signed == 0).astype(np.int16)
		mask = cv2.erode(mask, kernel)
		#mask = cv2.dilate(mask, dilate_kernel)

		self.current_stack_signed -= (mask == 0) * (self.current_stack_signed == 0)
		#cv2_og_visualize("Mask", mask)
		'''
		end_time = time.time()
		# rospy.loginfo("BATOLOPU process rate: " + str(1/(end_time-start_time)))
		
		'''

		# Adaptable weights
		a_ = 0.3
		b_ = 1 - a_
		batolopu = ((input_occ_grid + (input_occ_grid == 0) * self.current_stack_float) 
		            + (input_occ_grid == 1) * input_occ_grid).astype(np.float16)

		self.current_stack_float = (self.current_stack_unsigned.astype(np.float16) * a_ + batolopu * b_)
		self.current_stack_unsigned = (self.current_stack_float).astype(np.uint8)	
		'''

if __name__ == "__main__":
	rospy.init_node("stack_og")

	update_frequency = rospy.get_param("update_frequency", 10)
	fixed_frame = rospy.get_param("fixed_frame", "odom")
	baselink_frame = rospy.get_param("baselink_frame", "base_link")

	resolution = rospy.get_param("resolution", 0.1)
	height = rospy.get_param("height", 100)
	width = rospy.get_param("width", 100)

	threshold_value = rospy.get_param("threshold_value", 50)

	discard_time = rospy.get_param("discard_time", 120)

	topic_list = rospy.get_param("~/topic_list", ["/map/image_segmentation"])

	stack_og = OccupancyGridStack(fixed_frame, baselink_frame,
								  resolution, [height, width], topic_list)

	rate = rospy.Rate(1)
	while not rospy.is_shutdown():
		t1 = time.time()
		stack_og.publish_og_msg()
		t2 = time.time()
		runtime = t2-t1

		if runtime > 1/update_frequency:
			rospy.logwarn_throttle(1, "Stack grid did not achieve desired update rate: " + str(update_frequency) + ", actually took " + str(runtime))
		else:
			pass
			#time.sleep(1.0/update_frequency - runtime)


	


	

		