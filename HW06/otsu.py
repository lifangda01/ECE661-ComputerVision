#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt

def otsu_gray(image, nIter):
	'''
		Apply Otsu's algorithm on a gray scale image.
		@image: np.ndarray of input image
		@nIter: int of number of iterations to apply
		@return: np.ndarray of binary mask containing the foreground		
	'''
	mask = np.ones(image.shape).astype(np.uint8) * 255
	for i in xrange(nIter):
		hist,_ = np.histogram(image[np.nonzero(mask)], bins=np.arange(257), density=True)
		ihist = np.multiply(hist, np.arange(256))
		mu_T = np.sum(ihist)
		b_vars = np.zeros(256)
		for k in xrange(256):
			omega_k = np.sum(hist[:k])
			mu_k = np.sum(ihist[:k])
			if omega_k == 0.0 or omega_k == 1.0: continue
			b_vars[k] = (mu_T*omega_k - mu_k)**2 / omega_k / (1 - omega_k)
		k_star = np.argmax(b_vars)
		_,mask = cv2.threshold(image, k_star, 255, cv2.THRESH_BINARY)
	return mask

def otsu_rgb(image, nIters, invMasks):
	'''
		Apply Otsu's algorithm on a RGB image.
		@image: np.ndarray of input image
		@nIters: int of number of iterations to apply
		@invMasks: bool of whether to invert the mask
		@return: np.ndarray of segmented foreground
	'''
	# BGR channels
	channels = cv2.split(image)
	masks = []
	final_mask = np.ones(channels[0].shape).astype(np.uint8) * 255
	for i in range(len(channels)):
		mask = otsu_gray(channels[i], nIters[i])
		if invMasks[i]:
			mask = cv2.bitwise_not(mask)
		final_mask = cv2.bitwise_and(final_mask, mask)
		masks.append(mask)
	# Morphological operations to fill in holes and get rid of small regions
	final_mask = cv2.dilate(final_mask, np.ones((5,5)).astype(np.uint8))
	final_mask = cv2.erode(final_mask, np.ones((10,10)).astype(np.uint8))
	fig, axes = plt.subplots(2,3)
	for i in range(len(channels)):
		axes[0,i].set_aspect('equal')
		axes[0,i].imshow(channels[i], cmap='gray')
		axes[1,i].set_aspect('equal')
		axes[1,i].imshow(masks[i], cmap='gray')
	fig = plt.figure()
	plt.imshow(final_mask, cmap='gray')
	plt.show()
	return final_mask

def main():
	fpath1 = 'images/lake.jpg'
	color1 = cv2.imread(fpath1)
	final_mask = otsu_rgb(color1, [1,2,2], [0,1,1])


if __name__ == '__main__':
	main()