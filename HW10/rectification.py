#!/usr/bin/python
from pylab import *
import cv2
from scipy.optimize import leastsq

def apply_transformation_on_points(points, H):
	'''
		Apply the given transformation matrix on all points from input.
		@coords: list of input points, each is represented by (row,col)
		@return: list of points after transformation, each is represented by (row,col)
	'''
	l = []
	for point in points:
		p = array([point[0], point[1], 1.])
		p = dot(H, p)
		p = p / p[-1]
		l.append( (p[0], p[1]) )
	return l

def solve_homo_system_with_llsm(pts1, pts2):
	'''
		Obtain the the solution to a homogeneous linear least square problem.
		@pts1,pts2: list of matching points
		@return: ndarray of H		
	'''
	n = len(pts1)
	A = zeros((n,9))
	for i in range(n):
		x,y,w = pts1[i][0], pts1[i][1], 1. 
		xp,yp,wp = pts2[i][0], pts2[i][1], 1. 
		A[i] = [xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1]
	# Solution is the right eigenvector corresponding to the smallest eigenvalue
	U, s, Vt = svd( dot(A.T, A) )
	v = Vt[-1,:]
	return v

def condition_fundamental_matrix(F):
	'''
		Condition the fundamental matrix using SVD such that it's of rank 2.
	'''
	U, s, Vt = svd(F)
	s[-1] = 0.
	D = diag(s)
	F = dot( U, dot( D, Vt ) )
	F = F / F[-1,-1]
	return F

def get_normalization_homography_matrix():
	'''
		Find the homography matrix that transforms the given coordinates to zero-mean 
	'''
	pass

def get_right_null_space(A, eps=1e-5):
	'''
		Return the null space vector(s) of a matrix.
	'''
	U, s, Vt = svd(A)
	null_space = compress(s <= eps, Vt, axis=0)
	return null_space.T

def get_cross_product_equiv_matrix(w):
	'''
		Get the skew-symmetric cross-product equivalent matrix of a vector
	'''
	x,y,z = w[0],w[1],w[2]
	return array([[0., -z, y],
				[z, 0., -x],
				[-y, x, 0.]])

def triangulate_point(P, Pp, pt1, pt2):
	'''
		Given two corresponding points on the two image planes, return its physical coordinate.
	'''
	A = zeros((4,4))
	A[0,:] = pt1[0] * P[2,:] - P[0,:]
	A[1,:] = pt1[1] * P[2,:] - P[1,:]
	A[2,:] = pt2[0] * Pp[2,:] - Pp[0,:]
	A[3,:] = pt2[1] * Pp[2,:] - Pp[1,:]
	# Solution is the right eigenvector corresponding to the smallest eigenvalue
	U, s, Vt = svd( dot(A.T, A) )
	v = Vt[-1,:]
	return v / v[-1]

def triangulate_points(P, Pp, pts1, pts2):
	'''
		Convenience function.
	'''
	pts = []
	for pt1,pt2 in zip(pts1,pts2):
		pt = triangulate_point(P, Pp, pt1, pt2)
		pts.append(pt)
	return pts

def get_fundamental_matrix_from_projection(P, Pp):
	'''
		Extract F from the secondary canonical camera projection matrix.
	'''
	ep = Pp[:,3]
	s = get_cross_product_equiv_matrix(ep)
	F = dot( s, dot ( Pp, dot( P.T, inv( dot(P, P.T) ) ) ) )
	return F / F[-1,-1]

def nonlinear_optimization(pts1, pts2, P, Pp):
	'''
		Optimize the secondary camera matrix in canonical configuration.
	'''
	nPts = len(pts1)
	array_meas = hstack((array(pts1).T, array(pts2).T))
	array_reprj = zeros(array_meas.shape)
	p_guess = Pp.flatten()
	def error_function(p):
		'''
			Geometric distance as cost function for LevMar.
		'''
		Pp = p.reshape(3,4)
		array_reprj.fill(0.)
		for i in range(nPts):
			pt1, pt2 = pts1[i], pts2[i]
			pt_world = triangulate_point(P, Pp, pt1, pt2)
			pt1_reprj = dot(P, pt_world)
			pt1_reprj = pt1_reprj / pt1_reprj[-1]
			pt2_reprj = dot(Pp, pt_world)
			pt2_reprj = pt2_reprj / pt2_reprj[-1]
			array_reprj[:,i] = pt1_reprj[:2]
			array_reprj[:,i+nPts] = pt2_reprj[:2]
		error = array_meas - array_reprj
		return error.flatten()
	print "Optimizing..."
	p_refined, _ = leastsq(error_function, p_guess)
	Pp_refined = p_refined.reshape(3,4)
	Pp_refined = Pp_refined / Pp_refined[-1,-1]
	P_refined = P
	return P_refined, Pp_refined

def get_epipoles(F):
	'''
		Given fundamental matrix, return the left and right epipole.
	'''
	e = get_right_null_space(F)
	assert e.shape[1] == 1, "More than one left epipoles have been found."
	ep = get_right_null_space(F.T)
	assert ep.shape[1] == 1, "More than one right epipoles have been found."
	return e/e[-1], ep/ep[-1]

def get_canonical_projection_matrices(F, ep):
	'''
		Given fundamental matrix and epipole of the right image plane, find both camera projection matrices.
	'''
	P = hstack(( eye(3), zeros((3,1)) ))
	s = get_cross_product_equiv_matrix(ep)
	Pp = hstack(( dot(s, F), ep ))
	return P, Pp

def get_fundamental_matrix(pts1, pts2):
	'''
		Given point correspondences, make an initial estimate of fundamental matrix F.
	'''
	get_normalization_homography_matrix()
	f = solve_homo_system_with_llsm(pts1, pts2)
	F = f.reshape(3,3)
	F = condition_fundamental_matrix(F)
	return F

def get_rectification_homographies(image1, image2, pts1, pts2, e, ep, P, Pp):
	'''
		Find the homography matrices that align the corresponding epipolar lines to the same row.
	'''
	# Start with the second image first
	h2, w2 = image2.shape[0], image2.shape[1]
	# Translational matrix that shifts the image to be origin-centered
	T1 = array([[1., 0., -w2/2.], 
				[0., 1., -h2/2.],
				[0., 0., 1.]])
	# Rotational matrix that rotates the epipole onto x-axis
	theta = arctan( (ep[1] - h2/2.) / (ep[0] - w2/2.) )
	# Since we want to rotate to positive x-axis
	theta = -theta[0]
	R = array([[cos(theta), -sin(theta), 0.], 
				[sin(theta), cos(theta), 0.],
				[0., 0., 1.]])
	# Homography that takes epipole to infinity
	f = norm(array([ep[1] - h2/2., ep[0] - w2/2.]))
	G = array([[1., 0., 0.], 
				[0., 1., 0.],
				[-1./f, 0., 1.]])
	# Translate back to original center
	T2 = array([[1., 0., w2/2.], 
			[0., 1., h2/2.],
			[0., 0., 1.]])
	# The final homography for the second image
	Hp = dot( T2, dot( G, dot(R, T1) ) )
	####
	# Now the first image
	M = dot( Pp, dot( P.T, inv( dot(P, P.T) ) ) )
	H0 = dot( Hp, M )
	pts1h = apply_transformation_on_points(pts1, H0)
	pts2h = apply_transformation_on_points(pts2, Hp)
	# Construct inhomogeneous system
	n = len(pts1)
	A = zeros((n,3))
	b = zeros((n,1))
	for i in range(n):
		xh,yh = pts1h[i][0], pts1h[i][1] 
		xph = pts2h[i][0]
		A[i] = [xh, yh, 1.]
		b[i] = xph
	# h is pseudo-inverse multiplied by b
	h = dot( dot( inv( dot(A.T, A) ), A.T ), b )
	h = h.flatten()
	# Obtain the homography for the first image
	HA = array([[h[0], h[1], h[2]], 
			[0., 1., 0.],
			[0., 0., 1.]])
	H = dot(HA, H0)
	return H, Hp