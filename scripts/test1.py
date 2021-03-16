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

FIXED_FRAME     = "map"
BASE_LINK_FRAME = "base_link"
MAP_RESOLUTION  = 0.1 		#meter
OUTPUT_SIZE = (40,40) 		#meter
OUTPUT_SHAPE = tuple([int(round(z/MAP_RESOLUTION)) for z in OUTPUT_SIZE])
OCCUPIED_THRESHOLD  = 50

#===========================================================================================
# This section is for transformating things

def transform_data2sensor(og_np, translation, rotation): 
	global MAP_RESOLUTION
	global OUTPUT_SHAPE

	# Assume that we are doing 2D OccupancyGrid only
	theta_z             = np.arctan(rotation[2]/rotation[3])
	twod_quaternion     = [0, 0, np.sin(theta_z), np.cos(theta_z)]

	og_rot_mat          = ros_tf.transformations.quaternion_matrix(twod_quaternion)
	og_rot_mat          = og_rot_mat[0:2, 0:3]

	og_trans_mat        = np.zeros(2)
	og_trans_mat[0]     = OUTPUT_SHAPE[0]/2  + translation[0]/MAP_RESOLUTION
	og_trans_mat[1]     = OUTPUT_SHAPE[1]/2  + translation[1]/MAP_RESOLUTION

	warp_affine_mat       = og_rot_mat
	warp_affine_mat[:, 2] = og_trans_mat

	og_sensor_frame       = cv2.warpAffine(og_np, warp_affine_mat, OUTPUT_SHAPE)

	return og_sensor_frame

#-------------------------------------------------------------------------------------------

def transform_sensor2baselink(og_np, translation, rotation):
	global MAP_RESOLUTION
	global OUTPUT_SHAPE

	# Assume that we are doing 2D OccupancyGrid only
	theta_z    = np.arctan(rotation[2]/rotation[3]) * 2

	og_center  = tuple(np.array(og_np.shape[1::-1]) / 2)
	og_rot_mat = np.identity(3)
	og_rot_mat[0:2, :]    = cv2.getRotationMatrix2D(og_center, np.degrees(theta_z), 1.0)

	og_trans_mat   		  = np.identity(3)
	og_trans_mat[0:2,2]   = np.array([-translation[0], -translation[1]])/MAP_RESOLUTION

	warp_affine_mat       = np.matmul(og_rot_mat, og_trans_mat)

	warp_affine_mat       = warp_affine_mat[0:2, :]
	og_baselink_frame     = cv2.warpAffine(og_np, warp_affine_mat, OUTPUT_SHAPE)

	return og_baselink_frame

#-------------------------------------------------------------------------------------------

def transform_og2currentpose(og_np):
	global FIXED_FRAME
	global BASE_LINK_FRAME
	global OUTPUT_SHAPE
	global MAP_RESOLUTION
	
	# Everytime this function is used to create a new OG respects to it's temporal moment
	# the pose in that respecting moment is stored
	global last_current_pose

	og_center  = tuple(np.array(og_np.shape[1::-1]) / 2)

	#----------------------------------------------------------------------------------------
	# Calculate last and current pose transformation in FIXED frame

	last_translation, last_rotation = last_current_pose
	last_theta_z		     = np.arctan(last_rotation[2]/last_rotation[3]) * 2
	last_og_rot_mat 		 = np.identity(3)
	last_og_rot_mat[0:2, :]  = cv2.getRotationMatrix2D(og_center, np.degrees(last_theta_z), 1.0)

	last_og_trans_mat   	 = np.identity(3)
	last_og_trans_mat[0:2,2] = np.array([last_translation[1], -last_translation[0]])/MAP_RESOLUTION

	last_warp_affine_mat	 = np.matmul(last_og_trans_mat, last_og_rot_mat)

	#----------------------------------------------------------------------------------------

	current_translation, current_rotation    = tf_listener.lookupTransform(BASE_LINK_FRAME, 
																	       FIXED_FRAME, 
																		   rospy.Time(0))
	current_theta_z		        = np.arctan(current_rotation[2]/current_rotation[3]) * 2
	current_og_rot_mat 			= np.identity(3)
	current_og_rot_mat[0:2, :]  = cv2.getRotationMatrix2D(og_center, np.degrees(current_theta_z), 1.0)

	current_og_trans_mat   	    = np.identity(3)
	current_og_trans_mat[0:2,2] = np.array([current_translation[1], -current_translation[0]])/MAP_RESOLUTION

	current_warp_affine_mat	    = np.matmul(current_og_trans_mat, current_og_rot_mat)

	#----------------------------------------------------------------------------------------

	last_to_current_warp_affine_mat = np.matmul(np.linalg.inv(last_warp_affine_mat), current_warp_affine_mat)
	last_to_current_warp_affine_mat = last_to_current_warp_affine_mat[0:2, :]
	og_currentpose      				= cv2.warpAffine(og_np, last_to_current_warp_affine_mat, 
													 OUTPUT_SHAPE) 
	last_current_pose = (current_translation, current_rotation)

	return np.copy(og_currentpose)

#===========================================================================================
# This section is for OccupancyGrid stacking

def stack(current_stacked_og, baselink_transformed_og):
	#stacked = (og_current_pose + baselink_transformed_og*4)/5
	# batolopu: BAselink Transformed Occupancygrid LOw Priotizing Unknown
	# If new OccupancyGrid has unknown data, then unknown data is filled by 
	# stacked OccupancyGrid
	batolopu =  baselink_transformed_og + \
				(baselink_transformed_og==0).astype(np.int8) * current_stacked_og

	stacked_divided_og = (current_stacked_og + batolopu)/2

	return stacked_divided_og

#===========================================================================================
# This section is for receive new input OccupancyGrid and stack it

def callback(msg):
	global MAP_RESOLUTION
	global BASE_LINK_FRAME

	global stacked_og

	# Convert OccupancyGrid data to Numpy array
	map_w   = msg.info.width            #Unit: pixel
	map_h   = msg.info.height           #Unit: pixel 
	map_res = msg.info.resolution       #Unit: meter/pixel
	np_og   = np.array(msg.data).reshape(map_h, map_w) 

	# HUGE WARNING: 
	# Map to range 1-101, with 1 is unknown, 0 is free, 101 is occupied
	np_og   = (np_og+1).round().astype(np.uint8)

	# Resize Numpy OccupancyGrid data to the same global stack resolution
	resize_ratio = MAP_RESOLUTION/map_res
	resized_og   = cv2.resize(np_og, (0,0), fx=resize_ratio, fy=resize_ratio)

	#---------------------------------------------------------------------------------------
	# Transform OccupancyGrid to sensor's reference frame

	# Get data's origin's pose and position in sensor's reference frame
	sensor2origin_translation = [msg.info.origin.position.x,
								 msg.info.origin.position.y,
								 msg.info.origin.position.z]
	sensor2origin_rotation    = [msg.info.origin.orientation.x,
								 msg.info.origin.orientation.y,
								 msg.info.origin.orientation.z,
								 msg.info.origin.orientation.w]

	# Transform OccupancyGrid data to sensor's reference frame
	sensor_transformed_og = transform_data2sensor(resized_og, 
												  sensor2origin_translation,
												  sensor2origin_rotation)
	

	#---------------------------------------------------------------------------------------
	# Transform OccupancyGrid data to baselink's reference frame

	# Get static tf of sensor in baselink frame
	sensor_frame_id = msg.header.frame_id

	baselink2sensor_translation, baselink2sensor_rotation = tf_listener.lookupTransform(sensor_frame_id, 
																						BASE_LINK_FRAME, 
																						rospy.Time(0))

	baselink_transformed_og = transform_sensor2baselink(sensor_transformed_og,
														baselink2sensor_translation,
														baselink2sensor_rotation)

	#---------------------------------------------------------------------------------------
	# Transform old stacked OccupancyGrid to current pose
	# Then stack it
	transformed_stacked2currentpose = transform_og2currentpose(stacked_og)
	stacked_og = stack(transformed_stacked2currentpose, baselink_transformed_og)

	#cv2.imshow('data', cv2.flip(resized_og, 0))
	#cv2.imshow('sensor_frame', cv2.flip(sensor_transformed_og, 0))
	#cv2.imshow('baselink_frame', cv2.flip(baselink_transformed_og, 0))
	#cv2.waitKey(1)

#===========================================================================================
# This section is for getting message that's ready for ROS
def get_og_msg(og_np):
	global BASE_LINK_FRAME
	global OUTPUT_SIZE
	global OUTPUT_SHAPE
	global MAP_RESOLUTION

	cv2.imshow('og_np_input', og_np)
	# Subtract back to range [-1, 100]
	t0 = time.time()
	og_np = og_np.round().astype(np.int8) - 1
	t1 = time.time()
	cv2.imshow('og_np', (og_np+1)*100)
	occupancy_grid = og_np.flatten()
	occupancy_grid = occupancy_grid.tolist()

	map_msg = OccupancyGrid()

	map_msg.header = Header()
	map_msg.header.frame_id = BASE_LINK_FRAME
	map_msg.header.stamp    = rospy.Time.now()

	map_msg.info= MapMetaData()
	map_msg.info.height = OUTPUT_SHAPE[0]      #Unit: Pixel
	map_msg.info.width  = OUTPUT_SHAPE[1]      #Unit: Pixel
	map_msg.info.resolution = MAP_RESOLUTION

	map_msg.info.origin = Pose()
	map_msg.info.origin.position = Point()
	map_msg.info.origin.position.x = -OUTPUT_SIZE[0]/2      #Unit: Meter
	map_msg.info.origin.position.y = -OUTPUT_SIZE[1]/2      #Unit: Meter
	map_msg.info.origin.position.z = 0
	map_msg.info.origin.orientation = Quaternion()
	map_msg.info.origin.orientation.x = 0
	map_msg.info.origin.orientation.y = 0
	map_msg.info.origin.orientation.z = 0
	map_msg.info.origin.orientation.w = 1

	map_msg.data.extend(occupancy_grid)
	map_msg.info.map_load_time = rospy.Time.now()

	print(t1-t0)

	return map_msg

#===========================================================================================

# Note: unknown value (-1) is converted to 255 for cv2 and numpy to handle it
stacked_og = np.zeros(np.round(np.array(OUTPUT_SHAPE)/MAP_RESOLUTION).astype(np.uint8))

rospy.init_node('abc')
tf_listener = ros_tf.TransformListener()
time.sleep(1)

last_current_pose = tf_listener.lookupTransform('/map', 
  												'/base_link', 
												rospy.Time(0))

rospy.Subscriber('/map/image_segmentation', OccupancyGrid, callback, queue_size=1)
og_publisher = rospy.Publisher('/map/abc', OccupancyGrid, queue_size=1)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
	current_stacked_og = transform_og2currentpose(stacked_og)
	map_msg = get_og_msg(current_stacked_og)
	og_publisher.publish(map_msg)
	#print(map_msg)
	#cv2.imshow('current_stacked_og', cv2.flip(current_stacked_og, 0))
	cv2.waitKey(1)	
	rate.sleep()