#!/usr/bin/python
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

RESIZE_RATIO = 1.0

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def apply_filter(image, kernel):
	'''
		Convolve the image with filter.
		@image: np.ndarray of input image
		@kernel: np.ndarray of filter
		@return: np.ndarray of filtered image (new storage)
	'''
	return cv2.filter2D(image, -1, kernel)

def apply_haar_filter(image, sigma):
	'''
		Construct two Haar filter based on sigma,
		one along x direction, another along y direction.
		Then, apply the Haar filters on input image.
		@image: np.ndarray of input image
		@sigma: sigma of kernel
		@return: two np.ndarray images of gradients along row and column directions
	'''
	# Smallest even integer that is greater than 4*sigma
	s = int(4*sigma) + int(4*sigma)%2
	kernel_row = np.ones((s, s))
	kernel_col = np.ones((s, s))
	kernel_row[s/2:,:] = -kernel_row[s/2:,:]
	kernel_col[:,:s/2] = -kernel_col[:,:s/2]
	return apply_filter(image, kernel_row), apply_filter(image, kernel_col)

def get_covar_matrix(drow, dcol):
	'''
		Compute covariance matrix of an image patch.
		@drow, dcol: np.ndarray of gradients in the patch
		@return: the covariance matrix M
	'''
	# Point-wise multiplications
	drow2 = np.multiply(drow, drow)
	dcol2 = np.multiply(dcol, dcol)
	drowcol = np.multiply(drow, dcol)
	return np.array([
		[np.sum( np.sum(dcol2) ), np.sum( np.sum(drowcol) ) ],
		[np.sum( np.sum(drowcol) ), np.sum( np.sum(drow2) ) ]
		])

def get_corner_response(covar, k=0.04):
	'''
		Compute the corner reponse using eigen values of the covariance matrix.
		@covar: np.ndarray of input covariance matrix
		@return: the corner response number
	'''
	# Perform eigen decomposition to obtain eigen values
	eigens = np.linalg.eigvals(covar)
	lambda1, lambda2 = eigens[0], eigens[1]
	det = lambda1 * lambda2
	tr = lambda1 + lambda2
	return det - k*tr

def apply_nms():
	pass

def get_harris_corners(image, sigma, threshold):
	'''
		Find the corners in the input image using Harris corner detector.
		@image: np.ndarray of input image, double type
	'''
	start = time.time()
	corners = []
	(h, w) = image.shape
	drow_img, dcol_img = apply_haar_filter(image, 1.2)
	corner_img = np.zeros((h,w))
	# 5*sigma by 5*sigma neighboring window
	# Size should be always odd
	s = int(5*sigma) + (1 - int(5*sigma)%2)
	hs = s/2
	# Row means y, col means x
	for r in xrange(hs, h-hs):
		for c in xrange(hs, w-hs):
			M = get_covar_matrix(drow_img[r-hs:r+hs,c-hs:c+hs],
								 dcol_img[r-hs:r+hs,c-hs:c+hs])
			if np.linalg.matrix_rank(M, tol=10):
				corner_img[r,c] = 0
				continue
			R = get_corner_response(M)
			corner_img[r,c] = R
			if R > threshold:
				corners.append((r,c))
	print 'Corner detection took', time.time()-start, 'seconds'
	# plt.figure()
	# plt.imshow(corner_img, cmap='gray')
	# plt.title('Corner Response')
	# plt.show()
	return corners

def main():
	image = cv2.imread('images/pair3/1.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = np.double(image)
	# drow_img, dcol_img = apply_haar_filter(image, 1.2)
	# plt.subplot(1,3,1), plt.imshow(image, cmap='gray')
	# plt.subplot(1,3,2), plt.imshow(drow_img, cmap='gray')
	# plt.subplot(1,3,3), plt.imshow(dcol_img, cmap='gray')
	# plt.show()
	corners = get_harris_corners(image, 1.2, 150)
	(h, w) = image.shape
	fig, ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(image, cmap='gray')
	for corner in corners:
		ax.add_patch( Circle(corner, 5, fill=False, color=np.random.rand(3,1)) )
	# ax.title('Corner Response')
	plt.show()

if __name__ == '__main__':
	main()