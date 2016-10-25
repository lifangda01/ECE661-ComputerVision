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
	name = 'leopard'
	useTexture = True
	fpath = './images/' + name + '.jpg'
	if name == 'lake':
		# Lake
		nIters = [1,2,2]
		invMasks = [0,1,1]
		morphOps = [17,5]
		color = cv2.imread(fpath)
		if useTexture: 
			Ns = [3,5,7]
			nIters = [1,1,1]
			invMasks = [1,1,1]
			morphOps = [17,5]
			texture = get_texture_image(color, Ns)
			name = name + '_t'
	elif name == 'leopard':
		# Leopard
		nIters = [1,1,1]
		invMasks = [0,0,0]
		morphOps = [1,9]
		color = cv2.imread(fpath)
		if useTexture: 
			Ns = [5,7,9]
			nIters = [1,1,1]
			invMasks = [0,0,0]
			morphOps = [1,9]
			texture = get_texture_image(color, Ns)
			name = name + '_t'
	elif name == 'brain':
		# Brain
		nIters = [3,3,3]
		invMasks = [0,0,0]
		morphOps = [3,11]
		color = cv2.imread(fpath)
		if useTexture: 
			Ns = [5,7,9]
			nIters = [1,1,1]
			invMasks = [0,0,0]
			morphOps = [1,1]
			texture = get_texture_image(color, Ns)
			name = name + '_t'

	if useTexture:
		image = texture
	else:
		image = color
	channels = cv2.split(image)
	masks, final_mask = otsu_rgb(image, nIters, invMasks, morphOps)
	foreground = cv2.bitwise_and(color, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
	contour = get_contour(final_mask)
	# Save the images
	cv2.imwrite('./images/' + name + '_foreground.jpg', foreground)
	cv2.imwrite('./images/' + name + '_finalmask.jpg', final_mask)
	cv2.imwrite('./images/' + name + '_contour.jpg', contour)
	for i in range(len(channels)):
		cv2.imwrite('./images/' + name + '_mask_' + str(i) + '.jpg', masks[i])
	if useTexture:
		cv2.imwrite('./images/' + name + '_texture.jpg', texture)
	# Plots
	plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
	fig, axes = plt.subplots(2,3)
	for i in range(len(channels)):
		axes[0,i].set_aspect('equal')
		axes[0,i].imshow(channels[i], cmap='gray')
		axes[1,i].set_aspect('equal')
		axes[1,i].imshow(masks[i], cmap='gray')
	plt.figure()
	plt.imshow(final_mask, cmap='gray')
	plt.figure()
	plt.imshow(contour, cmap='gray')
	if useTexture:
		plt.figure()
		plt.imshow(texture)		
	plt.show()

if __name__ == '__main__':
	main()

	# color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
	# masks, final_mask = otsu_rgb(color, [2,1,1], [0,0,0])
	# cv2.imwrite('./images/' + name + '_foreground.jpg', cv2.cvtColor(foreground, cv2.COLOR_HSV2BGR))