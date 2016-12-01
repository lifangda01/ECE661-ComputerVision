#!/usr/bin/python
from pylab import *
import cv2
from sklearn.neighbors import NearestNeighbors

class LDAClassifier(object):
	def __init__(self, train_data, train_label, K):
		super(LDAClassifier, self).__init__()
		self.train_data = train_data
		self.train_label = train_label
		self.K = K
		self.m = None
		self.WK = None
		self.NN = None

	def train(self):
		'''
			Construct K-D tree based on the projections onto the subspace spanned by the K most discriminative vectors.
		'''
		print "======== LDA Training ========"
		pass

	def test(self, test_data, test_label):
		'''
			Project onto the K-D optimal subspace and find the nearest neighbor.
		'''
		print "======== LDA Testing ========"
		pass