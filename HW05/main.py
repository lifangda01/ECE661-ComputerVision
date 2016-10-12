#!/usr/bin/python
import numpy as np
import cv2
import ransac

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
	kp1, des1 = get_sift_kp_des(gray1)
	kp2, des2 = get_sift_kp_des(gray2)
	pts1, pts2 = get_matchings(kp1, des1, kp2, des2)
	ransac.apply_ransac_on_matchings(pts1, pts2, 0.1, 20)

def main():
	test_ransac()

if __name__ == '__main__':
	main()