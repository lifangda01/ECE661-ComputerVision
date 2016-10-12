#!/usr/bin/python
import numpy as np
import cv2

def get_inhomo_system(pts1, pts2):
	'''
		Construct A and b of the inhomogeneous equation system.
		@pts1,pts2: list of matching points
		@return: tuple of A and b
	'''
	n = len(pts1)
	A = np.zeros((2*n,8))
	b = np.zeros((2*n,1))
	for i in range(n):
		x,y,w = pts1[i][0], pts1[i][1], 1. 
		xp,yp,wp = pts2[i][0], pts2[i][1], 1. 
		A[2*i] = [0, 0, 0, -wp*x, -wp*y, -wp*w, yp*x, yp*y]
		A[2*i+1] = [wp*x, wp*y, wp*w, 0, 0, 0, -xp*x, -xp*y]
		b[2*i] = -yp*w
		b[2*i+1] = xp*w
	return A, b

def get_llsm_homograhpy_from_points(pts1, pts2):
	'''
		Obtain the homography using all the inliers from ransac.
		@pts1,pts2: list of matching points
		@return: np.ndarray of H		
	'''
	A, b = get_inhomo_system(pts1, pts2)
	# h is pseudo-inverse multiplied by b
	h = np.dot( np.dot( np.linalg.inv( np.dot(A.transpose(), A) ), A.transpose() ), b )
	h = np.append(h, 1.)
	h = h.reshape((3,3))
	return h