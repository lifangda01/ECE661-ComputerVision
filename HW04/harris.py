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

def apply_filter(image, kernel):
	'''
		Convolve the image with filter.
		@image: np.ndarray of input image
		@kernel: np.ndarray of filter
		@return: np.ndarray of filtered image (new storage)
	'''
	return cv2.filter2D(image, -1, kernel)

def get_integral_image(image):
	'''
		Compute the integral image of the input image.
		@image: np.ndarray of input image
		@return: np.ndarray of integral image
	'''
	return cv2.integral(image)

def apply_haar_filter(image, sigma):
	'''
		Construct two Haar filter based on sigma using convolution,
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

def apply_haar_filter_integral(int_img, sigma):
	'''
		Construct two Haar filter based on sigma using integral image,
		one along x direction, another along y direction.
		Then, apply the Haar filters on input image.
		@image: np.ndarray of input image
		@sigma: sigma of kernel
		@return: two np.ndarray images of gradients along row and column directions
	'''
	# Smallest even integer that is greater than 4*sigma
	s = int(4*sigma) + int(4*sigma)%2

	return apply_filter(image, kernel_row), apply_filter(image, kernel_col)

def get_covar_matrix(drow2, dcol2, drowcol):
	'''
		Compute covariance matrix of an image patch.
		@drow2, dcol2, drowcol: np.ndarray of precomputed images
		@return: the covariance matrix M
	'''
	return np.array([
		[np.sum( np.sum(dcol2) ), np.sum( np.sum(drowcol) ) ],
		[np.sum( np.sum(drowcol) ), np.sum( np.sum(drow2) ) ]
		])

def get_corner_response(covar, k=0.04):
	'''
		Compute the corner reponse using eigen values of the covariance matrix.
		@covar: np.ndarray of input covariance matrix
		@return: double of the corner response 
	'''
	# Perform eigen decomposition to obtain eigen values
	eigens = np.linalg.eigvals(covar)
	lambda1, lambda2 = eigens[0], eigens[1]
	det = lambda1 * lambda2
	tr = lambda1 + lambda2
	return det - k*tr*tr

def apply_nms(image, corners, win_size):
	'''
		Perform non-maximum compression on input corner image.
		@image: np.ndarray of corner image
		@corners: list of corners to be suppressed
		@win_size: int of the size of local window
		@return: list of suppressed corners
	'''
	(h, w) = image.shape
	hs = int(win_size/2)
	sup_corners = []
	for (r,c) in corners:
		if image[(r,c)] == np.max( image[r-hs:r+hs,c-hs:c+hs] ):
			sup_corners.append((r,c))
	return sup_corners

def get_harris_corners(image, sigma, threshold):
	'''
		Find the corners in the input image using Harris corner detector.
		@image: np.ndarray of input image, double type
		@sigma: double of scale
		@threshold: double of threshold for corner response
		@return: list of corners
	'''
	start = time.time()
	corners = []
	(h, w) = image.shape
	# 5*sigma by 5*sigma neighboring window
	# Size should be always odd
	s = int(5*sigma) + (1 - int(5*sigma)%2)
	hs = s/2
	# Blur the image to remove noise
	image = cv2.GaussianBlur(image, (s,s), 0)
	drow_img, dcol_img = apply_haar_filter(image, 1.2)
	# Preprocess the necessary images forms for computing covariance matrix
	drow2_img = np.multiply(drow_img, drow_img)
	dcol2_img = np.multiply(dcol_img, dcol_img)
	drowcol_img = np.multiply(drow_img, dcol_img)
	corner_img = np.zeros((h,w))
	print 'Starting corner detection...'
	# Row means y, col means x
	for r in xrange(hs, h-hs):
		for c in xrange(hs, w-hs):
			M = get_covar_matrix(drow2_img[r-hs:r+hs,c-hs:c+hs],
								 dcol2_img[r-hs:r+hs,c-hs:c+hs],
								 drowcol_img[r-hs:r+hs,c-hs:c+hs])
			R = get_corner_response(M)
			corner_img[r,c] = R
			if R > threshold:
				corners.append((r,c))
	corners = apply_nms(corner_img, corners, s)
	print 'Corner detection took', time.time()-start, 'seconds'
	# plt.figure()
	# plt.imshow(corner_img, cmap='gray')
	# plt.title('Corner Response')
	# plt.show()
	return corners

def main():
	ori = cv2.imread('images/pair3/1.jpg')
	ori = resize_image_by_ratio(ori, 1.0)
	image = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
	image = np.double(image) / 255.
	corners = get_harris_corners(image, 1.2, 100)
	fig, ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB), cmap='jet')
	for (y,x) in corners:
		ax.add_patch( Circle((x,y), 5, fill=False, color=np.random.rand(3,1), clip_on=False) )
	plt.show()
	# drow_img, dcol_img = apply_haar_filter(image, 1.2)
	# plt.subplot(1,3,1), plt.imshow(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB), cmap='jet')
	# plt.subplot(1,3,2), plt.imshow(drow_img, cmap='gray')
	# plt.subplot(1,3,3), plt.imsho1w(dcol_img, cmap='gray')
	# plt.show()

if __name__ == '__main__':
	main()