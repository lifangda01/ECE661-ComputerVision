#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from icp import icp

def read_depth_image(filename):
	'''
		Read the given .txt file into a float numpy array.
	'''
	with open(filename, 'r+') as f:
		rows = f.readlines()
	rows = [r.split() for r in rows]
	for i,r in enumerate(rows):
		rows[i] = map(float, r)
	return np.array(rows)

def depth_to_pc(dimg):
	'''
		Convert depth image into point cloud using the intrinsic camera matrix.
		@return: 3xN np.ndarray of points in the point cloud
	'''
	h,w = dimg.shape
	# Intrinsic camera matrix
	K = np.array([[365., 0., 256.],[0., 365., 212.],[0., 0., 1.]])
	K_inv = np.linalg.inv(K)
	# Preallocate memory for the point cloud
	npts = np.count_nonzero(dimg)
	pc = np.zeros((npts, 3))
	Y,X = np.nonzero(dimg)
	for i in range(npts):
		y,x = Y[i],X[i]
		u = np.array([x, y, 1])
		pc[i,:] = dimg[y,x] * np.dot(K_inv, u)
	return pc.transpose()

def read_pc_from_file(filename):
	'''
		Read the point cloud from file.
	'''
	dimg = read_depth_image(filename)
	print 'Reading depth image...', dimg.shape
	pc = depth_to_pc(dimg)
	print 'Converted to point cloud...', pc.shape
	return pc

def main():
	pc1 = read_pc_from_file('images/depthImage1ForHW.txt')
	pc2 = read_pc_from_file('images/depthImage2ForHW.txt')
	# Plot the two orginal point clouds
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.scatter(pc1[0,:], pc1[1,:], pc1[2,:], c='b', marker='.', edgecolor='none', depthshade=False)
	# ax.scatter(pc2[0,:], pc2[1,:], pc2[2,:], c='r', marker='.', edgecolor='none', depthshade=False)
	# plt.show()
	icp(pc1, pc2, 20)

if __name__ == '__main__':
	main()

	# plt.imshow(dimg1)
	# plt.savefig('./images/depth1.png')
	# plt.imshow(dimg2)
	# plt.savefig('./images/depth2.png')
	# plt.show()
