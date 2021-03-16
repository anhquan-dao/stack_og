#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import scipy
import scipy.ndimage
import skimage
import skimage.morphology

from nav_msgs.msg import OccupancyGrid

def cv2_og_visualize(name, og_np):
	show = og_np.astype(np.uint8)
	show[show == 1] = 255
	cv2.imshow(name, cv2.flip(show,0))
	cv2.waitKey(1)

def edgeDetection(msg):
	width = msg.info.width
	height = msg.info.height
	occupancy_matrix = np.array(msg.data).reshape(height, width)
	occupancy_matrix = (occupancy_matrix+1).round().astype(np.uint8)

	cv2_og_visualize("Input", occupancy_matrix)

	#threshold the occupancy grid
	occupancy_matrix_thresh = cv2.threshold(occupancy_matrix, 40, 100, cv2.THRESH_BINARY_INV)[1]
	occupancy_matrix_thresh[occupancy_matrix == 0] = 100

	cv2_og_visualize("Threshold", occupancy_matrix_thresh)

	kernel = np.ones((3,3), np.uint8)
	occupancy_matrix = cv2.morphologyEx(occupancy_matrix_thresh, cv2.MORPH_CLOSE, kernel)
	occupancy_matrix = cv2.blur(occupancy_matrix_thresh, (10,10))

	edge_images = cv2.Canny(occupancy_matrix_thresh, 50, 80)

	cv2_og_visualize("Canny detection", edge_images)

	distance_img = scipy.ndimage.distance_transform_edt(occupancy_matrix)
	morph_laplace_img = scipy.ndimage.morphological_laplace(distance_img, (10,10))
	midline = (morph_laplace_img < morph_laplace_img.min()/2).astype(np.uint8)
	midline= skimage.morphology.skeletonize(midline).astype(np.uint8)

	midline = cv2.dilate(midline, kernel, iterations = 1)

	occupancy_matrix[midline == 1] = 1
	cv2_og_visualize("Skeletonize", occupancy_matrix)

	cont = cv2.findContours(midline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	
	cv2.drawContours(occupancy_matrix, cont, -1, (255, 0, 0), 3)
	cv2_og_visualize("Contour", occupancy_matrix)
	

rospy.init_node("test_canny")

raw_stack_sub = rospy.Subscriber("output", OccupancyGrid, edgeDetection, queue_size = 1)

rospy.spin()