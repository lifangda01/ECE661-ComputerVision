#!/usr/bin/python
from pylab import *
import cv2
import os
from scipy.optimize import leastsq
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
	y0 = (W[0,1]*W[0,2] - W[0,0]*W[1,2]) / (W[0,0]*W[1,1] - W[0,1]**2)
	lamda = W[2,2] - ( W[0,2]**2 + y0*(W[0,1]*W[0,2] - W[0,0]*W[1,2]) ) / W[0,0]
	ax = sqrt(lamda / W[0,0])
	ay = sqrt(lamda*W[0,0] / (W[0,0]*W[1,1] - W[0,1]**2))
	s = - W[0,1]*ax*ax*ay / lamda
	x0 = s*y0 / ay - W[0,2]*ax*ax / lamda
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

def compensate_radial_distortion(arrayImage, x0, y0, k1, k2):
	'''
		Given points on image planes, compensate the radial distortion.
	'''
	r2 = (arrayImage[0,:]-x0)**2 + (arrayImage[1,:]-y0)**2
	r4 = r2**2
	arrayImage[0,:] = arrayImage[0,:] + (arrayImage[0,:]-x0) * (k1*r2 + k2*r4)
	arrayImage[1,:] = arrayImage[1,:] + (arrayImage[1,:]-y0) * (k1*r2 + k2*r4)
	return arrayImage

def reproject_corners(P, wcorners):
	'''
		Re-project pattern corners in world frame to image plane.
	'''
	# z in corners are zeros, i.e. corners are on z plane
	Hhat = P[:, [0,1,3]]
	pcorners = []
	for wcorner in wcorners:
		pcorner = dot(Hhat, wcorner)
		pcorner = pcorner / pcorner[-1]
		pcorners.append(pcorner)
	return pcorners

def pack_parameters(K, Rts):
	'''
		Given K and Rt matrices for all poses, pack them into a single vector.
	'''
	p = K.flatten()[[0,1,2,4,5]]
	for Rt in Rts:
		R = Rt[:,:3]
		r,_ = cv2.Rodrigues(R)
		r = r.flatten()
		t = Rt[:,3].flatten()
		p = hstack((p,r,t))
	return p

def unpack_parameters(p):
	'''
		Given a single vector, unpack into K and Rt matrices.
	'''
	K = array([[p[0], p[1], p[2]],
				[0., p[3], p[4]],
				[0., 0., 1.]])
	Rts = []
	p = p[5:].reshape(-1,6)
	nRows = p.shape[0]
	for i in range(nRows):
		row = p[i,:]
		R,_ = cv2.Rodrigues(row[:3])
		t = row[3:].reshape(3,-1)
		Rt = hstack((R,t))
		Rts.append(Rt)
	return K, Rts

def get_reprojection_accuracy(arrayImage, arrayReproj):
	'''
		Given array of repojected corners and ground truth corners, 
		return the mean and variance of the Euclidean distance between them.
	'''
	dist = norm( arrayImage - arrayReproj, axis=0 )
	return mean(dist), var(dist)

def calibrate_camera(dataset, N=40, d=25,levmar=True):
	wcorners = get_world_frame_corners(d)
	# Convert list to array, each column is a point
	arrayWorld = zeros((3,80))
	arrayImage = zeros((3,80*N))
	arrayReproj = zeros((3,80*N))
	for i,w in enumerate(wcorners): arrayWorld[:,i] = w
	Hs = []
	images = []
	print "Working on dataset...", dataset
	# dataset1
	if dataset == 1:
		# Compute all homography matrices 
		for i in range(1,N+1):
			image = imread('./Dataset1/Pic_{0}.jpg'.format(i))
			images.append(image)
			# Extract corners first
			icorners = extract_sorted_corners(image)
			arrayImage[:,(i-1)*80 : i*80] = array(icorners).T
			print 'Number of corners found...', len(icorners)
			# Compute the homography from world to image plane
			H = get_llsm_homograhpy_from_points(wcorners, icorners)
			Hs.append(H)
	# dataset2
	elif dataset == 2:
		# Compute all homography matrices 
		for i in range(N):
			image = imread('./Dataset2/{0}.jpg'.format(i))
			images.append(image)
			# Extract corners first
			icorners = extract_sorted_corners(image)
			arrayImage[:,i*80 : (i+1)*80] = array(icorners).T
			print 'Number of corners found...', len(icorners)
			# Compute the homography from world to image plane
			H = get_llsm_homograhpy_from_points(wcorners, icorners)
			Hs.append(H)
	# Extract the intrinsic matrix
	W = get_absolute_conic_image(Hs)
	K = get_intrinsic_matrix(W)
	# Extract the extrinsic matrix for each image
	Rts = []
	for H in Hs:
		Rt = get_extrinsic_matrix(K, H)
		Rts.append(Rt)
	for i,Rt in enumerate(Rts):
		P = dot(K, Rt)
		arrayReproj[:,i*80 : (i+1)*80] = array(reproject_corners(P, wcorners)).T
	print "==========Before LM=========="
	print "K =", K
	print "Euclidean distance: mean, variance =", get_reprojection_accuracy(arrayImage, arrayReproj)
	if not levmar: return K, Rts
	# Refine the K,R,t estimates with Levenberg-Marquedt
	p_guess = pack_parameters(K, Rts)
	def error_function(p):
		'''
			Geometric distance as cost function for LevMar.
		'''
		arrayTrans = zeros(arrayImage.shape)
		K, Rts = unpack_parameters(p)
		for i,Rt in enumerate(Rts):
			P = dot(K, Rt)
			Hhat = P[:, [0,1,3]]
			trans = dot(Hhat, arrayWorld)
			trans = trans / trans[-1,:]
			arrayTrans[:,i*80 : (i+1)*80] = trans
		error = arrayTrans - arrayImage
		return error[:2,:].flatten()
	print "Optimizing..."
	p_refined, _ = leastsq(error_function, p_guess)
	K_refined, Rts_refined = unpack_parameters(p_refined)
	for i,Rt_refined in enumerate(Rts_refined):
		P_refined = dot(K_refined, Rt_refined)
		arrayReproj[:,i*80 : (i+1)*80] = array(reproject_corners(P_refined, wcorners)).T
	print "==========After LM=========="
	print "K_refined =", K_refined
	print "Euclidean distance: mean, variance =", get_reprojection_accuracy(arrayImage, arrayReproj)
	return K_refined, Rts_refined

def main():
	d = 25
	# Obtain the camera matrix
	K, Rts = calibrate_camera(d=d, levmar=True)

if __name__ == '__main__':
	main()