#!/usr/bin/python
from pylab import *
import cv2
from sklearn.neighbors import NearestNeighbors

class LDAClassifier(object):
	def __init__(self):
		super(LDAClassifier, self).__init__()
		self.K = 0
		self.W = None
		self.NN = None

	def train(self, train_data, train_label, K):
		'''
			Construct K-D tree based on the projections onto the subspace spanned by the C-1 most discriminative vectors.
		'''
		print "======== LDA Training ========"
		self.train_data = train_data
		self.train_label = train_label
		self.K = K
		self.num_classes = unique(train_label).size
		# Follow Avi's notation
		X = self.train_data
		L = self.train_label
		C = self.num_classes
		if self.K > C-1: self.K = C-1
		# Get the class means and global mean
		m = mean(X, axis=1)
		M = zeros((X.shape[0],C))
		for i in range(1,C+1):
			M[:,i-1] = mean(X[:, L==i], axis=1)
		# Eigenvectors of the between class scatter are the same as mean difference vectors
		W = M-m.reshape(-1,1)
		W = W[:,:self.K]
		# Project all training samples to the K-D subspace
		Y = dot( W.T, X - m.reshape(-1,1) )
		# Construct K-D tree
		self.NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(Y.T)
		self.m = m
		self.W = W

	def test(self, test_data, test_label):
		'''
			Project onto the K-D optimal subspace and find the nearest neighbor.
		'''
		print "======== LDA Testing ========"
		num_test = test_label.size
		# Projection first
		X = test_data
		Y = dot( self.W.T, X - self.m.reshape(-1,1) )
		_, indices = self.NN.kneighbors(Y.T)
		pred = self.train_label[indices].flatten()
		accuracy = sum((pred - test_label) == 0) / float(num_test)
		print "Accuracy = {0:.2f}%".format(accuracy*100)
		return accuracy