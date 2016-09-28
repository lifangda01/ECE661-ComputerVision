from harris import get_harris_corners
from corresp import get_matching_SSD, get_matching_NCC
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def main():
	resize_ratio = 0.5
	sigma = 1.2
	threshold = 200
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
	# Plot image and mark the corners
	fig, axes = plt.subplots(1,2)
	axes[0].set_aspect('equal')
	axes[0].imshow(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB), cmap='jet')
	for (y,x) in corners1:
		axes[0].add_patch( Circle((x,y), 5, fill=False, color=np.random.rand(3,1), clip_on=False) )
	axes[1].set_aspect('equal')
	axes[1].imshow(cv2.cvtColor(color2, cv2.COLOR_BGR2RGB), cmap='jet')
	for (y,x) in corners2:
		axes[1].add_patch( Circle((x,y), 5, fill=False, color=np.random.rand(3,1), clip_on=False) )
	# Draw lines for matching pairs
	transFigure = fig.transFigure.inverted()
	coord1 = transFigure.transform(axes[0].transData.transform(corners1))
	coord2 = transFigure.transform(axes[1].transData.transform(corners2))
	lines = []
	for i in range(0, len(coord1)):
		pt1 = coord1[i]
		pt2 = coord2[i]
		line = Line2D((pt1[0],pt2[0]),(pt1[1],pt2[1]),transform=fig.transFigure)
		lines.append(line)
	fig.lines = lines
	plt.show()

if __name__ == '__main__':
	main()
