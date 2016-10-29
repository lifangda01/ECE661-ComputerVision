#!/usr/bin/python
import numpy as np
import cv2
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
	Q = data.transpose().astype(np.float32)
	P = model.transpose().astype(np.float32)
	# Initialize the K-D tree with model points 
	# FIXME: add threshold delta = 0.1?
	pNN = NearestNeighbors(n_neighbors=1).fit(P)
	for i in range(nIteration):
		# Find the NN pairs first
		_, indices = pNN.kneighbors(Q)
		# Get rid of the extra dimension in the indices
		indices = np.squeeze(indices)
		# Obtain the NN pairs
		Qp = Q
		Pp = P[indices,:]
		# print Qp.shape, Pp.shape
		# print Pp
		# print Qp
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
		C = np.dot(MP.transpose(), MQ)
		print C
		break

# def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
#     '''
#     The Iterative Closest Point estimator.
#     Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
#     their relative pose and the number of iterations
#     Returns the affine transform that transforms
#     the cloudpoint a to the cloudpoint b.
#     Note:
#         (1) This method works for cloudpoints with minor
#         transformations. Thus, the result depents greatly on
#         the initial pose estimation.
#         (2) A large number of iterations does not necessarily
#         ensure convergence. Contrarily, most of the time it
#         produces worse results.
#     '''

#     src = np.array([a.T], copy=True).astype(np.float32)
#     dst = np.array([b.T], copy=True).astype(np.float32)

#     #Initialise with the initial pose estimation
#     Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
#                    [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
#                    [0,                    0,                   1          ]])

#     src = cv2.transform(src, Tr[0:2])

#     for i in range(no_iterations):
#         #Find the nearest neighbours between the current source and the
#         #destination cloudpoint
#         nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
#         distances, indices = nbrs.kneighbors(src[0])

#         #Compute the transformation between the current source
#         #and destination cloudpoint
#         T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
#         #Transform the previous source and update the
#         #current source cloudpoint
#         src = cv2.transform(src, T)
#         #Save the transformation from the actual source cloudpoint
#         #to the destination
#         Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
#     return Tr[0:2]

# def main():
# 	#Create the datasets
# 	ang = np.linspace(-np.pi/2, np.pi/2, 320)
# 	a = np.array([ang, np.sin(ang)])
# 	th = np.pi/2
# 	rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
# 	b = np.dot(rot, a) + np.array([[0.2], [0.3]])

# 	#Run the icp
# 	M2 = icp(a, b, [0.1,  0.33, np.pi/2.2], 30)

# 	#Plot the result
# 	src = np.array([a.T]).astype(np.float32)
# 	res = cv2.transform(src, M2)
# 	plt.figure()
# 	plt.plot(b[0],b[1])
# 	plt.plot(res[0].T[0], res[0].T[1], 'r.')
# 	plt.plot(a[0], a[1])
# 	plt.show()

if __name__ == '__main__':
	main()