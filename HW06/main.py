#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from otsu import otsu_rgb

def get_texture_image(bgr, Ns):
	'''
		Represent the input image in texture space.
		@bgr: np.ndarray of input BGR image
		@Ns: list of int of window sizes
		@return: np.ndarray of image in texture space
	'''
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	texture = np.zeros( (bgr.shape[0], bgr.shape[1], len(Ns)) )
	for i, N in enumerate(Ns):
		w_h = int(N/2)
		for r in xrange(w_h, bgr.shape[0]-w_h):
			for c in xrange(w_h, bgr.shape[1]-w_h):
				var = np.var( gray[r-w_h:r+w_h+1, c-w_h:c+w_h+1] )
				texture[r,c,i] = var
	return texture.astype(np.uint8)

def get_contour(mask):
	'''
		Given an binary image, return a binary image that only contains the contour.
		@mask: np.ndarray of binary input image, i.e. 0 or 255
		@return: np.ndarray of binary contour image 
	'''
	contour = np.zeros(mask.shape).astype(np.uint8)
	w_h = 1
	for r in xrange(w_h, mask.shape[0]-w_h):
		for c in xrange(w_h, mask.shape[1]-w_h):
			if mask[r,c] == 0:
				continue
			if np.min( mask[r-w_h:r+w_h+1, c-w_h:c+w_h+1] ) == 0:
				contour[r,c] = 255
	return contour

def main():
	fpath1 = 'images/lake.jpg'
	color1 = cv2.imread(fpath1)

	# final_mask = otsu_rgb(color1, [1,2,2], [0,1,1])
	# foreground = cv2.bitwise_and(color1, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
	# plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
	# plt.figure()
	# plt.imshow(get_contour(final_mask), cmap='gray')
	# plt.show()

	color1 = cv2.GaussianBlur(color1, (3,3), 0)
	texture1 = get_texture_image(color1, [5,7,9])
	final_mask = otsu_rgb(texture1, [1,1,1], [1,1,1])
	final_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
	foreground = cv2.bitwise_and(color1, final_mask)
	plt.figure()
	plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
	plt.show()

if __name__ == '__main__':
	main()
