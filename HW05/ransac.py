#!/usr/bin/python
import numpy as np
import cv2

def get_N(epsilon, n, p):
	'''
		Calculate the number of trials, N.
		@epsilon: double of probability of outlier
		@n: int of size of minimal set
		@p: double of probability that at least one out of N trials only contain inlier
		@return: int of number of trials
	'''
	return int( np.log(1 - p) / np.log( 1 - (1 - epsilon)**n ) )

def get_M(epsilon, n_total):
	'''		
		Calculate minimum number of inliers, M.
		@epsilon: double of probability of outlier
		@n_total: int of total number of samples
		@return: int of minimum number of inliers
	'''
	return int( (1 - epsilon) * n_total )

def get_random_minimal_set(n, pts1, pts2):
	'''		
		Randomly pick and copy n matchings.
		@n: int of size of minimal set
		@pts1,pts2: list of points of all the matchings
		@return: tuple of lists of n matchings
	'''
	perm = np.random.permutation(len(pts1))
	s1 = []
	s2 = []
	for i in range(n):
		s1.append( pts1[ perm[i] ] )
		s2.append( pts2[ perm[i] ] )
	return s1, s2

def get_homography(pts1, pts2):
	'''
		Calculate the homography matrix H based on matchings.
		@matchings: list of matchings
		@return: np.ndarray of H
	'''
	pts1 = np.float32(pts1)
	pts2 = np.float32(pts2)
	return cv2.getPerspectiveTransform(pts1, pts2)

def apply_homography_on_points(points, H):
	'''
		Apply H on all the given points.
		@points: list of points
		@H: np.ndarray of H
		@return: list of points after transformation
	'''
	points = np.array( [np.float32(points)] )
	return np.squeeze( cv2.perspectiveTransform(points, H) )

def get_inliers(trans1, pts1, pts2, delta):
	'''
		Get number of inliers based on Euclidean distance and delta
		@trans1,pts1,pts2: list of points
		@delta: double of threshold
		@return: int of number of inliers
	'''
	count = 0
	in1 = []
	in2 = []
	for i in range(len(trans1)):
		pt1, pt2 = trans1[i], pts2[i]
		dist = np.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )
		if dist < delta:
			count += 1
			in1.append( pts1[i] )
			in2.append( pts2[i] )
	return count, in1, in2

def apply_ransac_on_matchings(pts1, pts2, epsilon, delta):
	'''
		Use RANSAC to find best matchings.
	'''
	N = get_N(epsilon, 4, 0.99)
	M = get_M(epsilon, len(pts1))
	print "M =", M, "; N =", N
	for i in range(N):
		ms1, ms2 = get_random_minimal_set(4, pts1, pts2)
		H = get_homography(ms1, ms2)
		trans1 = apply_homography_on_points(pts1, H)
		num_inliers, in1, in2 = get_inliers(trans1, pts1, pts2, delta)
		if num_inliers > M:
			print "Number of inliers =", num_inliers
			return in1, in2
	print "ERROR: RANSAC has failed to find a valid minimal set..."
	return [], []


def main():
	pass

if __name__ == '__main__':
	main()