#!/usr/bin/python
from pylab import *
import cv2

def extract_homo_lines_from_image(image):
	'''
		Given a checkerboard input image, return the detected lines.
	'''
	height, width = image.shape[0], image.shape[1]
	# Blur the image to remove background edges
	# 7 by 7 kernel, default sigma
	blurred = cv2.GaussianBlur(image, (7,7), 0)
	# Apply Canny edge
	# minVal and maxVal
	edges = cv2.Canny(blurred, 100, 200)
	# Probabilistic Hough transform
	# distance resolution, angle resolution and accumulator threshold
	lines = cv2.HoughLinesP(edges, 1, pi/180, 25, minLineLength=100, maxLineGap=100)
	lines = squeeze(lines)
	print "Number of lines found before refinement...", lines.shape[0]
	# Draw the lines
	# for x1,y1,x2,y2 in lines:
	# 	hline = cross( array([x1,y1,1]), array([x2,y2,1]) )
	# 	a,b,c = hline
	# 	cv2.line(image, (0, -c/b), (width-1,-(c+a*(width-1))/b), color=(0,255,0), thickness=2)
	# imshow(edges, cmap='gray')
	# figure()
	# imshow(image)
	# show()
	return [cross( array([x1,y1,1]), array([x2,y2,1]) ) for x1,y1,x2,y2 in lines]

def extract_sorted_corners_from_homo_lines(homolines):
	'''
		Given 'vertical' and 'horizontal' lines, return their intersections.
	'''
	homolines = array(homolines)
	# We need to divide the input lines into two groups, horizontal and vertical
	theta = array([arctan(float(-a)/(b+0.01)) for a,b,c in homolines])
	horilines = homolines[logical_and(theta >= -pi/4, theta < pi/4), :]
	vertlines = homolines[logical_or(theta < -pi/4, theta >= pi/4), :]
	# Now we need to sort the lines for future labeling process
	Yinters = -horilines[:,2] / horilines[:,1]
	horilines = horilines[ argsort(Yinters), :]
	Xinters = -vertlines[:,2] / vertlines[:,0]
	vertlines = vertlines[ argsort(Xinters), :]
	# Find the intersections in order
	homocorners = []
	for hline in horilines:
		for vline in vertlines:
			curr = cross(hline, vline)
			curr = curr / curr[2]
			# Skip this corner if it's too close to one of the previous corner
			if min([norm(curr-prev) for prev in homocorners] + [15]) >= 15:
				homocorners.append(curr)
	return homocorners

def extract_sorted_corners(image):
	homolines = extract_homo_lines_from_image(image)
	return extract_sorted_corners_from_homo_lines(homolines)

def main():
	n = 40
	for i in range(1,n+1):
		image = imread('./Dataset1/Pic_{0}.jpg'.format(i))
		homolines = extract_homo_lines_from_image(image)
		corners = extract_sorted_corners_from_homo_lines(homolines)
		print './Dataset1/Pic_{0}.jpg'.format(i), 'Number of corners found', len(corners)
	
	# image = imread('./Dataset1/Pic_{0}.jpg'.format(18))
	# height, width = image.shape[0], image.shape[1]
	# homolines = extract_homo_lines_from_image(image)
	# corners = extract_sorted_corners_from_homo_lines(homolines)
	# for a,b,c in homolines:
	# 	cv2.line(image, (0, -c/b), (width-1,-(c+a*(width-1))/b), color=(0,255,0), thickness=1)
	# for i,corner in enumerate(corners):
	# 	cv2.circle(image, (corner[0], corner[1]), 2, color=(0,0,255), thickness=1)
	# 	cv2.putText(image, str(i), (corner[0]-10, corner[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,0), thickness=1)
	# imshow(image)
	# show()


if __name__ == '__main__':
	main()