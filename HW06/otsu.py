#!/usr/bin/python
import numpy as np
import cv2

def otsu_gray(image, nIter):
	'''
		Apply Otsu's algorithm on a grayscale image.
		@image: np.ndarray of input image
		@return: np.ndarray of segmented image		
	'''
	k = 1, k_star = 1
	hist,_ = np.histogram(image, bins=np.arange(256), density=True)
	ihist = np.multiply(hist, np.arange(256))
	mu_T = np.sum(ihist)
	for i in xrange(nIter):
		for k in xrange(256):
			
		omega_k = np.sum(hist[:k])
		mu_k = np.sum(ihist[:k])
		sigmaB2_k = (mu_T*omega_k - mu_k)**2 / omega_k / (1 - omega_k)


def otsu_rgb(image):
	'''
		Apply Otsu's algorithm on a RGB image.
		@image: np.ndarray of input image
		@return: np.ndarray of segmented image
	'''
	# BGR channels
	channels = cv2.split(image)
	# for C in channels:

def main():
	fpath1 = 'images/lake.jpg'
	color1 = cv2.imread(fpath1)
	

if __name__ == '__main__':
	main()