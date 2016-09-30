from harris import get_harris_corners
from corresp import get_matching_SSD, get_matching_NCC
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def main():
	resize_ratio = 1.0
	sigma = 1.2
	threshold = 300
	# Find corners in the first image
	color1 = cv2.imread('images/pair3/1.jpg')
	color1 = resize_image_by_ratio(color1, resize_ratio)
	gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
	gray1 = np.double(gray1) / 255.
	corners1 = get_harris_corners(gray1, sigma, threshold)
	# Find corners in the second image
	color2 = cv2.imread('images/pair3/2.jpg')
	color2 = resize_image_by_ratio(color2, resize_ratio)
	gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	gray2 = np.double(gray2) / 255.
	corners2 = get_harris_corners(gray2, sigma, threshold)
	# Get the matchings
	s = int(5*sigma) + (1 - int(5*sigma)%2)
	corners1, corners2 = get_matching_SSD(gray1, corners1, gray2, corners2, s)
	# corners1, corners2 = get_matching_NCC(gray1, corners1, gray2, corners2, s)
	# Don't forget to convert r,c format to x,y
	corners1 = [(c[1], c[0]) for c in corners1]
	corners2 = [(c[1], c[0]) for c in corners2]
	# Plot image and mark the corners
	fig, axes = plt.subplots(1,2)
	axes[0].set_aspect('equal')
	axes[0].imshow(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB), cmap='jet')
	axes[1].set_aspect('equal')
	axes[1].imshow(cv2.cvtColor(color2, cv2.COLOR_BGR2RGB), cmap='jet')
	for i in range(0, len(corners1)):
		color = np.random.rand(3,1)
		pt1 = corners1[i]
		pt2 = corners2[i]
		axes[0].add_patch( Circle(pt1, 5, fill=False, color=color, clip_on=False) )
		axes[1].add_patch( Circle(pt2, 5, fill=False, color=color, clip_on=False) )
		# Draw lines for matching pairs
		line1 = ConnectionPatch(xyA=pt1, xyB=pt2, coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1], color=color)
		line2 = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA='data', coordsB='data', axesA=axes[1], axesB=axes[0], color=color)
		axes[0].add_patch(line1)
		axes[1].add_patch(line2)
	plt.show()

if __name__ == '__main__':
	main()
