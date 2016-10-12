#!/usr/bin/python
import numpy as np
import cv2
from ransac import apply_ransac_on_matchings
from llsm import get_llsm_homograhpy_from_points
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def get_sift_kp_des(image, nfeatures=0):
	'''
		Extract SIFT key points and descriptors from image.
		@image: np.ndarray of input image, double type, gray scale
		@return: tuple of key points and corresponding descriptors
	'''
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
	kps, descs = sift.detectAndCompute(image, None)
	print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
	return kps, descs

def get_matchings(kp1, des1, kp2, des2):
	'''
		Get matchings using SIFT and return non-openCV points
	'''
	bf = cv2.BFMatcher()
	matches = bf.match(des1, des2)
	pts1, pts2 = [], []
	for m in matches:
		pt1 = kp1[m.queryIdx].pt
		pt2 = kp2[m.trainIdx].pt
		pts1.append(pt1)
		pts2.append(pt2)
	return pts1, pts2	

def test_ransac():
	fpath1 = 'images/1.jpg'
	fpath2 = 'images/2.jpg'
	resize_ratio = 0.5
	color1 = cv2.imread(fpath1)
	color1 = resize_image_by_ratio(color1, resize_ratio)
	gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
	color2 = cv2.imread(fpath2)
	color2 = resize_image_by_ratio(color2, resize_ratio)
	gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	kp1, des1 = get_sift_kp_des(gray1, nfeatures=1000)
	kp2, des2 = get_sift_kp_des(gray2, nfeatures=1000)
	pts1, pts2 = get_matchings(kp1, des1, kp2, des2)
	in1, in2 = apply_ransac_on_matchings(pts1, pts2, 0.9, 20)
	print get_llsm_homograhpy_from_points(in1, in2)
	# Plot image and mark the corners
	fig, axes = plt.subplots(1,2)
	axes[0].set_aspect('equal')
	axes[0].imshow(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB), cmap='jet')
	axes[1].set_aspect('equal')
	axes[1].imshow(cv2.cvtColor(color2, cv2.COLOR_BGR2RGB), cmap='jet')
	for i in range(len(pts1)):
		color = np.random.rand(3,1)
		pt1 = pts1[i]
		pt2 = pts2[i]
		axes[0].add_patch( Circle(pt1, 5, fill=False, color=color, clip_on=False) )
		axes[1].add_patch( Circle(pt2, 5, fill=False, color=color, clip_on=False) )
		# Draw lines for matching pairs
		line1 = ConnectionPatch(xyA=pt1, xyB=pt2, coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1], color=color)
		line2 = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA='data', coordsB='data', axesA=axes[1], axesB=axes[0], color=color)
		axes[0].add_patch(line1)
		axes[1].add_patch(line2)
	plt.show()

def show_sift():
	fpath1 = 'images/1.jpg'
	fpath2 = 'images/2.jpg'
	resize_ratio = 0.5
	color1 = cv2.imread(fpath1)
	color1 = resize_image_by_ratio(color1, resize_ratio)
	gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
	color2 = cv2.imread(fpath2)
	color2 = resize_image_by_ratio(color2, resize_ratio)
	gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	kp1, des1 = get_sift_kp_des(gray1, nfeatures=1000)
	kp2, des2 = get_sift_kp_des(gray2, nfeatures=1000)
	# Find the matchings
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append([m])
	# Weird fix for cv2.drawMatchesKnn error
	img3 = np.zeros((1,1))
	# cv2.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB),kp1,
							cv2.cvtColor(color2, cv2.COLOR_BGR2RGB),kp2,
							good,img3,flags=2)
	plt.imshow(img3),plt.show()

def main():
	test_ransac()
	# show_sift()


if __name__ == '__main__':
	main()