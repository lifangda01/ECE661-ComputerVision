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

def get_matching_SSD(image1, points1, image2, points2, win_size):
	'''
		Find the matching pairs using SSD metric.
		@image1,image2: np.ndarray of two input gray level images
		@points1,points2: list of interest points
		@win_size: int of size of the neighboring window
		@return: two lists of corresponding points (same index)
	'''
	l1 = points1
	l2 = []
	hs = int(win_size/2)
	for pt1 in points1:
		p1 = get_patch(image1, pt1, win_size)
		SSDs = [SSD( p1 , get_patch(image1, pt2, win_size) ) for pt2 in points2]
		l2.append( points2[np.argmin(SSDs)])
	return l1, l2

def get_matching_NCC(image1, points1, image2, points2, win_size):
	'''
		Find the matching pairs using NCC metric.
		@image1,image2: np.ndarray of two input gray level images
		@points1,points2: list of interest points
		@image1,image2: np.ndarray of two input gray level images
		@return: two lists of corresponding points (same index)
	'''
	l1 = points1
	l2 = []
	hs = int(win_size/2)
	for pt1 in points1:
		p1 = get_patch(image1, pt1, win_size)
		NCCs = [NCC( p1 , get_patch(image1, pt2, win_size) ) for pt2 in points2]
		l2.append( points2[np.argmax(NCCs)])
	return l1, l2

if __name__ == '__main__':
	main()