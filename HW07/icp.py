#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

def icp(data, model, nIteration):
	'''
		Find the transformation matrix that transforms data to model.
		@data: np.ndarray of data point cloud, Mx3
		@model: np.ndarray of model point cloud, Nx3
		@nIteration: int of ICP iterations
		@return: np.ndarray of transformation matrix, 4x4
	'''
	T = np.zeros((4,4))
	Q = data.astype(np.float32)
	P = model.astype(np.float32)
	Qt = Q
	# Initialize the K-D tree with model points 
	# FIXME: add threshold delta = 0.1?
	pNN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(P)
	for i in range(nIteration):
		# Find the NN pairs first
		_, indices = pNN.kneighbors(Qt)
		# Get rid of the extra dimension in the indices
		indices = np.squeeze(indices)
		print 'indices', indices
		# Obtain the NN pairs
		Qp = Qt
		Pp = P[indices,:]
		# Find the centroids
		Qc = np.sum(Qp, axis=0) / Qp.shape[0]
		Pc = np.sum(Pp, axis=0) / Pp.shape[0]
		print 'Qc', Qc, 'Pc', Pc
		# Subtract the centroid from the point pairs
		MQ = Qp - Qc
		MP = Pp - Pc
		# Compute the correlation matrix
		C = np.dot(MQ.transpose(), MP)
		print 'C', C
		# Decompose C using SVD and compute rotation and translation
		U,s,V = np.linalg.svd(C)
		print U.shape, V.shape, s.shape
		R = np.dot(V.transpose(),U.transpose())
		t = Pc.transpose() - np.dot(R, Qc.transpose())
		# Construct 4x4 transformation matrix T
		T[:3,:3] = R
		T[:3,3] = t
		T[3,3] = 1.
		print 'T', T
		# Transform our original data with transformation matrix
		Qt = np.dot( T, np.transpose( np.hstack(( Qp, np.ones((Qp.shape[0],1)) )) ) )
		Qt = Qt[:3,:].transpose()
		print Qt.shape
	return Qt
