#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

def icp(data, model, nIteration):
	'''
		Find the transformation matrix that transforms data to model.
		@data: np.ndarray of data point cloud, 3xM
		@model: np.ndarray of model point cloud, 3xN
		@nIteration: int of ICP iterations
		@return: np.ndarray of transformation matrix, 4x4
	'''
	T = np.zeros((4,4))
	Q = data.transpose().astype(np.float32)
	P = model.transpose().astype(np.float32)
	Qt = np.copy(Q)
	# Initialize the K-D tree with model points 
	# FIXME: add threshold delta = 0.1?
	pNN = NearestNeighbors(n_neighbors=1).fit(P)
	for i in range(nIteration):
		# Find the NN pairs first
		_, indices = pNN.kneighbors(Qt)
		# Get rid of the extra dimension in the indices
		indices = np.squeeze(indices)
		# Obtain the NN pairs
		Qp = Qt
		Pp = P[indices,:]
		# Find the centroids
		Qc = np.sum(Qp, axis=0) / Qp.shape[0]
		Pc = np.sum(Pp, axis=0) / Pp.shape[0]
		# Subtract the centroid from the point pairs
		MP = Pp - Pc
		MQ = Qp - Qc
		print MP.shape, MQ.shape
		# Compute the correlation matrix
		C = np.dot(MQ.transpose(), MP)
		print C
		# Decompose C using SVD and compute rotation and translation
		U,s,V = np.linalg.svd(C)
		R = np.dot(V,U.transpose())
		t = Pc.transpose() - np.dot(R, Qc.transpose())
		# Construct 4x4 transformation matrix T
		T[:3,:3] = R
		T[:3,3] = t
		T[3,3] = 1.
		# Transform our original data with transformation matrix
		Qt = np.dot( T, np.transpose( np.concatenate( Qp, np.ones((Qp.shape[0],1)) ) ) )
		print Qt
	return Qt		
