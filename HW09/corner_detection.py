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
	lines = cv2.HoughLinesP(edges, 1, pi/180, 25, minLineLength=100, maxLineGap=600)
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

def remove_duplicate_lines(homolines):
	homolines = homolines / homolines[:,1]
	unique = [homolines[0]]
	for curr in homolines[1:]:
		prev = unique[-1]
		# if norm(curr-prev)


def extract_sorted_corners_from_homo_lines(homolines):
	'''
		Given 'vertical' and 'horizontal' lines, return their intersections.
	'''
	homolines = array(homolines)
	# We need to divide the input lines into two groups, horizontal and vertical
	theta = array([arctan(float(-a)/b) for a,b,c in homolines])
	horilines = homolines[theta < pi/4, :]
	vertlines = homolines[pi/4 <= theta, :]
	# Now we need to sort the lines for future labeling process
	Yinters = -horilines[:,2] / horilines[:,1]
	horilines = horilines[ argsort(Yinters), :]
	Xinters = -vertlines[:,2] / vertlines[:,0]
	vertlines = vertlines[ argsort(Xinters), :]
	# Find the intersections in order
	homocorners = []
	# numVlines = vertlines.shape[0]
	for hline in horilines:
		for vline in vertlines:
			curr = cross(hline, vline)
			curr = curr / curr[2]
			if len(homocorners) > 0:
				prevleft = homocorners[-1]
			else:
				prevleft = array([0,0,1])
			if len(homocorners) > 8:
				prevtop = homocorners[-8]
			else:
				prevtop = array([0,0,1])
			# Skip this corner if it's too close to the previous corner
			# print norm(curr-prev), prev, curr, norm(curr-prev)>10
			if norm(curr-prevleft) > 10 and norm(curr-prevtop) > 10:
				homocorners.append(curr)
	print len(homocorners)

	image = imread('./Dataset1/Pic_10.jpg')
	height, width = image.shape[0], image.shape[1]
	for a,b,c in horilines:
		cv2.line(image, (0, -c/b), (width-1,-(c+a*(width-1))/b), color=(0,255,0), thickness=1)
	for a,b,c in vertlines:
		cv2.line(image, (0, -c/b), (width-1,-(c+a*(width-1))/b), color=(255,0,0), thickness=1)
	for i,homocorner in enumerate(homocorners):
		cv2.circle(image, (homocorner[0], homocorner[1]), 2, color=(0,0,255), thickness=1)
		cv2.putText(image, str(i), (homocorner[0]-10, homocorner[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,0), thickness=1)
	imshow(image)
	show()

def main():
	img = imread('./Dataset1/Pic_10.jpg')
	homolines = extract_homo_lines_from_image(img)
	corners = extract_sorted_corners_from_homo_lines(homolines)

if __name__ == '__main__':
	main()