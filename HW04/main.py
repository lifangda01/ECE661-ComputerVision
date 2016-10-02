from harris import get_harris_corners
from metric import get_matching_SSD, get_matching_NCC
from sift import get_sift_kp_des
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def find_matching(fpath1, fpath2, feature, metric, resize_ratio, sigma, threshold, match_win_size):
	'''
		Generic wrapper function for finding matches between interest points in two images.
		@fpath1,fpath2: string of path to the two RGB image files
		@feature: string of feature to use ('Harris', 'SURF', 'SIFT')
		@metric: string of metric to use ('NCC', 'SSD')
		@resize_ratio: double of the actual size of the image to use
		@sigma: double of scale
		@threshold: double of threshold for corners response in Harris
		@match_win_size: int of size of the matching window in metric
	'''
	color1 = cv2.imread(fpath1)
	color1 = resize_image_by_ratio(color1, resize_ratio)
	gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
	color2 = cv2.imread(fpath2)
	color2 = resize_image_by_ratio(color2, resize_ratio)
	gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	if feature == 'Harris':
		# We need double precision
		gray1 = np.double(gray1) / 255.
		gray2 = np.double(gray2) / 255.
		# Find corners in the first image
		features1 = get_harris_corners(gray1, sigma, threshold)
		# Find corners in the second image
		features2 = get_harris_corners(gray2, sigma, threshold)
		# Get the matchings
		if metric == 'SSD':
			features1, features2 = get_matching_SSD(gray1, features1, gray2, features2, match_win_size)
		elif metric == 'NCC':
			features1, features2 = get_matching_NCC(gray1, features1, gray2, features2, match_win_size)
		else:
			print "ERROR: Unknown metric..."
			return
		# Don't forget to convert r,c format to x,y
		features1 = [(c[1], c[0]) for c in features1]
		features2 = [(c[1], c[0]) for c in features2]
		# Plot image and mark the corners
		fig, axes = plt.subplots(1,2)
		axes[0].set_aspect('equal')
		axes[0].imshow(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB), cmap='jet')
		axes[1].set_aspect('equal')
		axes[1].imshow(cv2.cvtColor(color2, cv2.COLOR_BGR2RGB), cmap='jet')
		for i in range(0, len(features1)):
			color = np.random.rand(3,1)
			pt1 = features1[i]
			pt2 = features2[i]
			axes[0].add_patch( Circle(pt1, 5, fill=False, color=color, clip_on=False) )
			axes[1].add_patch( Circle(pt2, 5, fill=False, color=color, clip_on=False) )
			# Draw lines for matching pairs
			line1 = ConnectionPatch(xyA=pt1, xyB=pt2, coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1], color=color)
			line2 = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA='data', coordsB='data', axesA=axes[1], axesB=axes[0], color=color)
			axes[0].add_patch(line1)
			axes[1].add_patch(line2)
		plt.show()
	elif feature == 'SIFT':
		# Find the keypoints and descriptor in one go
		kp1, des1 = get_sift_kp_des(gray1)
		kp2, des2 = get_sift_kp_des(gray2)
		# Find the matchings
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2, k=2)
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
   		return

def main():
	# find_matching('images/pair3/1.jpg', 'images/pair3/2.jpg', 'Harris', 'SSD',
	# 				1.0, 1.2, 100, 20)
	# find_matching('images/pair3/1.jpg', 'images/pair3/2.jpg', 'Harris', 'NCC',
	# 				1.0, 1.2, 100, 20)
	find_matching('images/pair3/1.jpg', 'images/pair3/2.jpg', 'SIFT', None,
				1.0, None, None, None)	
if __name__ == '__main__':
	main()
