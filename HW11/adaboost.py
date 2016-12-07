#!/usr/bin/python
from pylab import *
import cv2

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
			
		def _train():
			pass

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

