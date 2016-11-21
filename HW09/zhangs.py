#!/usr/bin/python
from pylab import *
import cv2
import os
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

def get_extrinsic_matrix(K, H):
	'''
		Given K and H for a camera pose,
		return the extrinsic matrix [R|t].
	'''
	h1, h2, h3 = H[:,0], H[:,1], H[:,2]
	Kinv = inv(K)
	epsilon = 1. / norm(dot(Kinv, h1))
	r1 = epsilon * dot(Kinv, h1)
	r2 = epsilon * dot(Kinv, h2)
	r3 = cross(r1, r2)
	t = epsilon * dot(Kinv, h3)
	R = vstack((r1,r2,r3)).T
	# Condition the rotation matrix to be orthonormal
	U, _, Vt = svd(R)
	R = dot(U, Vt)
	return hstack((R,t.reshape(3,1)))

def reproject_corners(P, wcorners):
	'''
		Reproject pattern corners in world frame to image plane.
	'''
	# z in corners are zeros, i.e. corners are on z plane
	Hhat = P[:, [0,1,3]]
	pcorners = []
	for wcorner in wcorners:
		pcorner = dot(Hhat, wcorner)
		pcorner = pcorner / pcorner[-1]
		pcorners.append(pcorner)
	return pcorners

def main():
	N = 40	# number of images
	d = 25	# size of square in mm
	wcorners = get_world_frame_corners(d)
	Hs = []
	images = []
	# Compute all homography matrices 
	for i in range(1,N+1):
		image = imread('./Dataset1/Pic_{0}.jpg'.format(i))
		images.append(image)
		# Extract corners first
		icorners = extract_sorted_corners(image)
		print './Dataset1/Pic_{0}.jpg'.format(i), 'Number of corners found...', len(icorners)
		# Compute the homography from world to image plane
		H = get_llsm_homograhpy_from_points(wcorners, icorners)
		Hs.append(H)
	# Extract the intrinsic matrix
	W = get_absolute_conic_image(Hs)
	K = get_intrinsic_matrix(W)
	print "K =", K
	# Extract the extrinsic matrix for each image
	Rts = []
	for H in Hs:
		Rt = get_extrinsic_matrix(K, H)
		Rts.append(Rt)

	# Backproject to validate
	n = 10
	Rt = Rts[n]
	P = dot(K, Rt)
	print "Rt =", Rt
	print "P =", P
	pcorners = reproject_corners(P, wcorners)
	image = images[n]
	height, width = image.shape[0], image.shape[1]
	corners = extract_sorted_corners(image)
	for i in range(len(corners)):
		pcorner = pcorners[i].astype(int)
		corner = corners[i]
		cv2.circle(image, (corner[0], corner[1]), 2, color=(0,0,255), thickness=1)
		cv2.circle(image, (pcorner[0], pcorner[1]), 2, color=(255,0,0), thickness=1)
		cv2.putText(image, str(i), (corner[0]-10, corner[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,0), thickness=1)
	imshow(image)
	show()

if __name__ == '__main__':
	main()