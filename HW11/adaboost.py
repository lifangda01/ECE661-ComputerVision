#!/usr/bin/python
from pylab import *
import cv2

WIDTH = 40
HEIGHT = 20
NUMFEATURES = 88935

def get_integral_image(image):
	'''
		Compute and return the integral representation of an gray-scale image.
	'''
	return cumsum(cumsum(image,axis=0),axis=1)

def _extract_features(intimg):
	'''
		Return a feature vector for a given image.
	'''
	features = zeros(NUMFEATURES)
	count = 0
	# Do horizontal ones first, 1x2 base size
	bW, bH = 2, 1
	for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
		for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
			for y in range(0, HEIGHT - bH*i + 1):	# Iterate through the image 
				for x in range(0, WIDTH - bW*j + 1):
					features[count] = intimg[y,x] - 2*intimg[y,x+bW*j/2] \
									+ intimg[y,x+bW*j] - intimg[y+bH*i,x] \
									+ 2*intimg[y+bH*i,x+bW*j/2] - intimg[y+bH*i,x+bW*j]
					count = count + 1
	bW, bH = 1, 2
	for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
		for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
			for y in range(0, HEIGHT - bH*i + 1):	# Iterate through the image 
				for x in range(0, WIDTH - bW*j + 1):
					features[count] = -intimg[y,x] + intimg[y,x+bW*j] \
									+ 2*intimg[y+bH*i/2,x] - 2*intimg[y+bH*i/2,x+bW*j] \
									- intimg[y+bH*i,x] + intimg[y+bH*i,x+bW*j]
					count = count + 1
	print "Total number of features...", count
	return features

def _get_feature_matrix():
	'''
		Return the feature matrix.
	'''
	features = zeros((NUMFEATURES, WIDTH*HEIGHT))
	count = 0
	# Do horizontal ones first, 1x2 base size
	bW, bH = 2, 1
	for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
		for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
			for y in range(0, HEIGHT - bH*i + 1):	# Iterate through the image 
				for x in range(0, WIDTH - bW*j + 1):
					features[count, y*HEIGHT+x] = 1.0
					features[count, y*HEIGHT+x+bW*j/2] = -2.0
					features[count, y*HEIGHT+x+bW*j] = 1.0
					features[count, (y+bH*i)*HEIGHT+x] = -1.0
					features[count, (y+bH*i)*HEIGHT+x+bW*j/2] = 2.0
					features[count, (y+bH*i)*HEIGHT+x+bW*j] = -1.0
					count = count + 1
	bW, bH = 1, 2
	for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
		for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
			for y in range(0, HEIGHT - bH*i + 1):	# Iterate through the image 
				for x in range(0, WIDTH - bW*j + 1):
					features[count, y*HEIGHT+x] = -1.0
					features[count, y*HEIGHT+x+bW*j] = 1.0
					features[count, (y+bH*i/2)*HEIGHT+x] = 2.0
					features[count, (y+bH*i/2)*HEIGHT+x+bW*j] = -2.0
					features[count, (y+bH*i)*HEIGHT+x/2] = -1.0
					features[count, (y+bH*i)*HEIGHT+x+bW*j] = 1.0
					count = count + 1
	print "Total number of features...", count
	return features

def extract_features(data):
	'''
		Extract all features from integral images into a nFeatures x nSamples array.
	'''
	# We are only concerned with two-rectangle edge features
	# Get the sorted list of sample indices.
	feature_matrix = _get_feature_matrix()
	feature_vectors = dot( feature_matrix, data )
	return feature_vectors

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

def main():
	extract_features()

if __name__ == '__main__':
	main()