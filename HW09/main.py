#!/usr/bin/python
from pylab import *
import cv2
import os
from corner_detection import extract_sorted_corners
from zhangs import calibrate_camera, reproject_corners, get_world_frame_corners

def test_dataset1():
	d = 25
	N = 40
	dataset = 1
	# Obtain the camera matrix
	K, Rts = calibrate_camera(dataset, N=N, d=d, levmar=True)
	# Backproject to validate
	n = 30
	Rt = Rts[n-1]
	print "Rt =", Rt
	P = dot(K, Rt)
	image = imread('./Dataset1/Pic_{0}.jpg'.format(n))
	height, width = image.shape[0], image.shape[1]
	wcorners = get_world_frame_corners(d)
	pcorners = reproject_corners(P, wcorners)
	corners = extract_sorted_corners(image)
	for i in range(len(corners)):
		pcorner = pcorners[i].astype(int)
		corner = corners[i]
		cv2.circle(image, (corner[0], corner[1]), 2, color=(0,0,255), thickness=1)
		cv2.circle(image, (pcorner[0], pcorner[1]), 2, color=(255,0,0), thickness=1)
		cv2.putText(image, str(i), (corner[0]-10, corner[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,0), thickness=1)
	imshow(image)
	# imsave("./figures/reproject_3.png",image)
	show()

def test_dataset2():
	d = 25
	N = 20
	dataset = 2
	# Obtain the camera matrix
	K, Rts = calibrate_camera(dataset, N=N, d=d, levmar=True)
	# Backproject to validate
	n = 15
	Rt = Rts[n]
	print "Rt =", Rt
	P = dot(K, Rt)
	image = imread('./Dataset2/{0}.jpg'.format(n))
	height, width = image.shape[0], image.shape[1]
	wcorners = get_world_frame_corners(d)
	pcorners = reproject_corners(P, wcorners)
	corners = extract_sorted_corners(image)
	for i in range(len(corners)):
		pcorner = pcorners[i].astype(int)
		corner = corners[i]
		cv2.circle(image, (corner[0], corner[1]), 4, color=(0,0,255), thickness=2)
		cv2.circle(image, (pcorner[0], pcorner[1]), 4, color=(255,0,0), thickness=2)
		cv2.putText(image, str(i), (corner[0]-10, corner[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,0), thickness=1)
	imshow(image)
	# imsave("./figures/reproject_4.png",image)
	show()

def main():
	test_dataset1()
	test_dataset2()

if __name__ == '__main__':
	main()