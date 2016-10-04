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

def get_patch(image, point, win_size):
	'''
		Convenient function to return the patch centered at point of a certain size.
	'''
	hs = int(win_size/2)
	r = point[0]
	c = point[1]
	return image[r-hs:r+hs,c-hs:c+hs]

def check_boundary(image, points, win_size):
	'''
		Given the size of window, return the points whose window does not cross boundary.
	'''
	hs = int(win_size/2)
	return [pt for pt in points if hs <= pt[0] < image.shape[0]-hs 
								and hs <= pt[1] < image.shape[1]-hs]


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
	enum = np.sum( np.sum( np.multiply( (p1-m1), (p2-m2) ) ) )
	denom = np.sum( np.sum( np.square(p1-m1) ) ) * np.sum( np.sum( np.square(p2-m2) ) )
	return enum / np.sqrt(denom)

def get_matching_SSD(image1, points1, image2, points2, win_size):
	'''
		Find the matching pairs using SSD metric.
		@image1,image2: np.ndarray of two input gray level images
		@points1,points2: list of interest points
		@win_size: int of size of the neighboring window
		@return: two lists of corresponding points (same index)
	'''
	minInd = []
	minVal = []
	# Perform boundary check
	points1 = check_boundary(image1, points1, win_size)
	points2 = check_boundary(image2, points2, win_size)
	for pt1 in points1:
		p1 = get_patch(image1, pt1, win_size)
		SSDs = [SSD( p1 , get_patch(image2, pt2, win_size) ) for pt2 in points2]
		idx = np.argmin(SSDs)
		minInd.append( idx )
		minVal.append( SSDs[idx] )
	# Threshold based on minimum SSD
	absmin = np.min(minVal)
	threshold = absmin * 5.
	l1 = []
	l2 = []
	for i in range(0, len(points1)):
		if minVal[i] < threshold:
			l1.append( points1[ i ] )
			l2.append( points2[ minInd[i] ] )	
	return l1, l2

def get_matching_NCC(image1, points1, image2, points2, win_size):
	'''
		Find the matching pairs using NCC metric.
		@image1,image2: np.ndarray of two input gray level images
		@points1,points2: list of interest points
		@image1,image2: np.ndarray of two input gray level images
		@return: two lists of corresponding points (same index)
	'''
	maxInd = []
	maxVal = []
	# Perform boundary check
	points1 = check_boundary(image1, points1, win_size)
	points2 = check_boundary(image2, points2, win_size)
	for pt1 in points1:
		p1 = get_patch(image1, pt1, win_size)
		NCCs = [NCC( p1 , get_patch(image2, pt2, win_size) ) for pt2 in points2]
		idx = np.argmax(NCCs)
		maxInd.append( idx )
		maxVal.append( NCCs[idx] )
	# Threshold based on maximum NCC
	absmax = np.max(maxVal)
	threshold = absmax * 0.9
	l1 = []
	l2 = []
	for i in range(0, len(points1)):
		if maxVal[i] > threshold:
			l1.append( points1[ i ] )
			l2.append( points2[ maxInd[i] ] )	
	return l1, l2

if __name__ == '__main__':
	main()
