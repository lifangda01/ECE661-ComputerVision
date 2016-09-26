#!/usr/bin/python
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def SSD(p1, p2):
	'''
		Calculate the SSD between two input image patches.
		@p1,p2: np.ndarray of two input patches
		@return: double of SSD
	'''
	return np.sum( np.sum( np.square(p1 - p2) ) )

def NCC(p1, p2):
	'''
		Calculate the NCC between two input image patches.
		@p1,p2: np.ndarray of two input patches
		@return: double of NCC
	'''
	m1 = np.mean(p1)
	m2 = np.mean(p2)
	enum = np.sum( np.sum( (p1-m1)(p2-m2) ) )
	denom = np.sum( np.sum( np.square(p1-m1) ) ) * np.sum( np.sum( np.square(p2-m2) ) )
	return enum / np.sqrt(denom)

def get_matching_SSD(image1, points1, image2, points2):
	'''
		Find the matching pairs using SSD metric.
		@image1,image2: np.ndarray of two input gray level images
		@return: two lists of corresponding points (same index)
	'''
	pass

def get_matching_NCC(image1, points1, image2, points2):
	'''
		Find the matching pairs using SSD metric.
		@image1,image2: np.ndarray of two input gray level images
		@return: two lists of corresponding points (same index)
	'''
	pass

if __name__ == '__main__':
	main()