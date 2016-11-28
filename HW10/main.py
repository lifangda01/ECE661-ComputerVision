#!/usr/bin/python
from pylab import *
import cv2
from rectification import *

def pick_points(image):
	'''
		Prompt user to click points on image.
	'''
	points = []
	fig = plt.figure()
	def onclick(event):
		if len(points) == 8:
			return
		x = int(event.xdata)
		y = int(event.ydata)
		points.append((x,y))
		print "Mouse clicked at (x,y)...", (x,y), "; Total number of points clicked...", len(points)
		print points
	imshow(image)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	show()

def main():
	image1 = imread('./images/1.jpg')
	image2 = imread('./images/2.jpg')
	# subplot(1,2,1)
	# imshow(image1)
	# subplot(1,2,2)
	# imshow(image2)
	# Uncomment if necessary
	# pick_points(image1)
	# pick_points(image2)
	pts1 = [(92, 112), (290, 142), (131, 263), (245, 296), (151, 326), (151, 365), (101, 493), (277, 428)]
	pts2 = [(39, 142), (257, 115), (66, 259), (176, 310), (80, 315), (81, 352), (52, 441), (248, 503)]
	F = get_fundamental_matrix(pts1, pts2)
	print "F = ", F
	e, ep = get_epipoles(F)
	print "e = ", e
	print "ep = ", ep
	P, Pp = get_canonical_projection_matrices(F, ep)
	print "P = ", P
	print "Pp = ", Pp
	H, Hp =	get_rectification_homographies(image1, image2, pts1, pts2, e, ep, P, Pp)
	print "H = ", H
	print "Hp = ", Hp
	print "-------- After Rectification --------"
	print "e = ", dot(H, e)
	print "ep = ", dot(Hp, ep)

	rectified1 = cv2.warpPerspective(image1, H, (1000,1000))
	rectified2 = cv2.warpPerspective(image2, Hp, (1000,1000))
	figure()
	subplot(1,2,1)
	imshow(rectified1)
	subplot(1,2,2)
	imshow(rectified2)
	show()

if __name__ == '__main__':
	main()