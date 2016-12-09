#!/usr/bin/python
from pylab import *
import cv2

WIDTH = 40
HEIGHT = 20

def _get_integral_image(image):
	'''
		Compute and return the integral representation of an gray-scale image.
	'''
	return cumsum(cumsum(image,axis=0),axis=1)

def _define_features():
	'''
		Return a feature matrix where each row is a feature vector.
	'''
	# Do horizontal ones first
	for s in range()

def extract_features(data):
	'''
		Extract all features into a nFeatures x nSamples array.
	'''
	# Iterate through all features
	# We are only concerned with two-rectangle edge features
	# Get the sorted list of sample indices.
	pass

class CascadedAdaBoostClassifier(object):
	def __init__(self):
		super(CascadedAdaBoostClassifier, self).__init__()

		def train(self, train_data, train_label, f, d, F_target):
			'''
				Train cascaded AdaBoost classifiers given user-defined:
				max acceptable fpr per layer f, min acceptable detection rate per layer d,
				and overall fpr F_target.
			'''
			self.train_data = train_data
			self.train_label = train_label
			

class AdaBoostClassifier(object):
	def __init__(self):
		super(AdaBoostClassifier, self).__init__()

	def set_training_dataset():
		pass

	def add_weak_classifier():
		'''
			Add the current best weak classifier on the weighted training set.
		'''
		pass

	def evaluate():
		'''
			Evaluate the strong classifier's performance given a decision threshold.
		'''
		pass

