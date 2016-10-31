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
		@return: Nx3 np.ndarray of points in the point cloud
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
	return pc

def read_pc_from_file(filename):
	'''
		Read the point cloud from file.
		Each point is stored as a row vector.
	'''
	dimg = read_depth_image(filename)
	print 'Reading depth image...', dimg.shape
	pc = depth_to_pc(dimg)
	print 'Converted to point cloud...', pc.shape
	return pc

def main():
	# Each point is a row vector
	data = read_pc_from_file('images/depthImage1ForHW.txt')
	model = read_pc_from_file('images/depthImage2ForHW.txt')

	# Toy PCs for debugging
	# x = np.linspace(0, 2*np.pi, 100)
	# y = np.zeros(100)
	# z1 = np.sin(x)
	# z2 = np.sin(x) + 0.2
	# data = np.vstack((x,y,z1)).transpose()
	# model = np.vstack((x,y,z2)).transpose()

	# Plot the two orginal point clouds
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c='b', marker='.', edgecolor='none', depthshade=False)
	ax.scatter(model[:,0], model[:,1], model[:,2], c='r', marker='.', edgecolor='none', depthshade=False)
	ax.view_init(elev=16, azim=-52)
	# plt.savefig('./images/before.png')
	plt.show()
	aligned = icp(data, model, 20)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(aligned[:,0], aligned[:,1], aligned[:,2], c='b', marker='.', edgecolor='none', depthshade=False)
	ax.scatter(model[:,0], model[:,1], model[:,2], c='r', marker='.', edgecolor='none', depthshade=False)
	ax.view_init(elev=16, azim=-52)
	# plt.savefig('./images/after.png')
	plt.show()

if __name__ == '__main__':
	main()