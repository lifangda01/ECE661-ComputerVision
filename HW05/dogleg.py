#!/usr/bin/python
import numpy as np
import cv2
import numdifftools as nd

pts1 = []
pts2 = []

def apply_transformation_on_points(points, H):
	'''
		Apply the given transformation matrix on all points from input.
		@coords: list of input points, each is represented by (row,col)
		@return: list of points after transformation, each is represented by (row,col)
	'''
	l = []
	for point in points:
		p = np.array([point[0], point[1], 1.])
		p = np.asarray(np.dot(H, p)).reshape(-1)
		p = p / p[-1]
		l.append( (p[0], p[1]) )
	return l

def get_epsilon(H):
	'''
		Get X-f(p).
	'''
	pts1p = apply_transformation_on_points(pts1, H)
	nPts = len(pts1p)
	epsilon = np.array( (nPts, 1) )
	for i in range(nPts):
		pt1, pt2 = pts1p[i], pts2[i]
		epsilon[i] = np.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )
	return epsilon

def C(H_list):
	'''
		The cost function.
	'''
	H_list.append(1.)
	H = np.array(H_list).reshape(3,3)
	epsilon = get_epsilon(H)
	return np.dot(epsilon.transpose(), epsilon)

def get_jacobian(H):
	'''
		Return Jacobian matrix.
	'''
	H_list = list(H.reshape(-1)[:-1])
	return nd.Jacobian(C)(H_list)

def get_delta_GD():
	'''
		Calculate the next step by Gradient Descent/
	'''
	pass

def main():
	pass

if __name__ == '__main__':
	main()