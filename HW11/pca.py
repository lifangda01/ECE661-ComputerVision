#!/usr/bin/python
from pylab import *
import cv2
from sklearn.neighbors import NearestNeighbors

class PCAClassifier(object):
	def __init__(self):
		super(PCAClassifier, self).__init__()
		self.K = 0
		self.m = None
		self.WK = None
		self.NN = None

	def train(self, train_data, train_label, K):
		'''
			Construct K-D tree based on the projections onto the subspace spanned by the first K eigenvectors of the covariance matrix.
		'''
		print "======== PCA Training ========"
		self.train_data = train_data
		self.train_label = train_label
		self.K = K
		# Follow the notation of Avi's tutorial
		X = self.train_data
		m = mean(X, axis=1)
		_, _, Ut = svd(dot(X.T, X))
		W = dot(X, Ut.T)
		# Preserve the first K eigenvectors
		WK = W[:,:self.K]
		# Project all training samples to the K-D subspace
		Y = dot( WK.T, X - m.reshape(-1,1) )
		# Construct K-D tree
		self.NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(Y.T)
		self.m = m
		self.WK = WK

	def test(self, test_data, test_label):
		'''
			Project onto the K-D optimal subspace and find the nearest neighbor.
		'''
		print "======== PCA Testing ========"
		num_test = test_label.size
		# Projection first
		X = test_data
		Y = dot( self.WK.T, X - self.m.reshape(-1,1) )
		_, indices = self.NN.kneighbors(Y.T)
		pred = self.train_label[indices].flatten()
		accuracy = sum((pred - test_label) == 0) / float(num_test)
		print "K = {0}; Accuracy = {1:.2f}%".format(self.K, accuracy*100)
		return accuracy