#!/usr/bin/python
from pylab import *
import cv2
from llsm import get_llsm_homograhpy_from_points
from corner_detection import extract_sorted_corners

def get_vij(H, i, j):
	'''
		Construct one row of V.
	'''
	# i=0,1,2; j=0,1,2
	H = H.T
	return array([ H[i,0]*H[j,0],
				H[i,0]*H[j,1] + H[i,1]*H[j,0],
				H[i,1]*H[j,1],
				H[i,2]*H[j,0] + H[i,0]*H[j,2],
				H[i,2]*H[j,1] + H[i,1]*H[j,2],
				H[i,2]*H[j,2] ])

def get_absolute_conic_image(Hs):
	'''
		Given homography matrices,
		obtain W, image of the absolute conic.
	'''
	numH = len(Hs)
	W = zeros((3,3))
	V = zeros((2*numH, 6))
	for k,H in enumerate(Hs):
		V[2*k] = get_vij(H, 0, 1)
		V[2*k+1] = get_vij(H, 0, 0) - get_vij(H, 1, 1)
	# b is the eigenvector of V for the smallest eigenvalue
	u, s, vt = svd(dot( V.T, V ))
	b = vt[-1,:]
	W = array([ [b[0], b[1], b[3]],
				[b[1], b[2], b[4]],
				[b[3], b[4], b[5]] ])
	return W

def get_intrinsic_matrix(W):
	'''
		Given the image of absolute conic W,
		return the intrinsic camera matrix K.
	'''
	x0 = (W[0,1]*W[0,2] - W[0,0]*W[1,2]) / (W[0,0]*W[1,1] - W[0,1]**2)
	lamda = W[2,2] - ( W[0,2]**2 + x0*(W[0,1]*W[0,2] - W[0,0]*W[1,2]) ) / W[0,0]
	ax = sqrt(lamda / W[0,0])
	ay = sqrt(lamda*W[0,0] / (W[0,0]*W[1,1] - W[0,1]**2))
	s = - W[0,1]*ax*ax*ay / lamda
	y0 = s*x0 / ay - W[0,2]*ax*ax / lamda
	K = array([ [ax, s, x0],
				[0., ay, y0],
				[0., 0., 1.] ])
	return K

def get_world_frame_corners(d):
	'''
		Given the size of squares on the pattern,
		return the coordinates of all the corners in world frame.
	'''
	corners = []
	for i in range(10):
		for j in range(8):
			corner = array([j*d, i*d, 1])
			corners.append(corner)
	return corners

def main():
	n = 40	# number of images
	d = 25	# size of square in mm
	wcorners = get_world_frame_corners(d)
	Hs = []
	# Compute all homography matrices 
	for i in range(1,n+1):
		image = imread('./Dataset1/Pic_{0}.jpg'.format(i))
		# Extract corners first
		icorners = extract_sorted_corners(image)
		print './Dataset1/Pic_{0}.jpg'.format(i), 'Number of corners found', len(icorners)
		# Compute the homography from world to image plane
		H = get_llsm_homograhpy_from_points(wcorners, icorners)
		Hs.append(H)
	W = get_absolute_conic_image(Hs)
	K = get_intrinsic_matrix(W)
	print "W = ", W
	print "K = ", K

if __name__ == '__main__':
	main()