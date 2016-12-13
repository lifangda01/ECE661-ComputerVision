#!/usr/bin/python
from pylab import *
import cv2

WIDTH = 40
HEIGHT = 20
NUMFEATURES = 88935 # Obtained by _get_feature_matrix()

def get_integral_image(image):
	'''
		Compute and return the integral representation of an gray-scale image.
	'''
	return cumsum(cumsum(image,axis=0),axis=1)

# def _extract_features(intimg):
	# '''
	# 	Return a feature vector for a given image.
	# '''
	# features = zeros(NUMFEATURES)
	# count = 0
	# # Do horizontal ones first, 1x2 base size
	# bW, bH = 2, 1
	# for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
	# 	for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
	# 		for y in range(0, HEIGHT - bH*i + 1):	# Iterate through the image 
	# 			for x in range(0, WIDTH - bW*j + 1):
	# 				features[count] = intimg[y,x] - 2*intimg[y,x+bW*j/2] \
	# 								+ intimg[y,x+bW*j] - intimg[y+bH*i,x] \
	# 								+ 2*intimg[y+bH*i,x+bW*j/2] - intimg[y+bH*i,x+bW*j]
	# 				count = count + 1
	# bW, bH = 1, 2
	# for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
	# 	for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
	# 		for y in range(0, HEIGHT - bH*i + 1):	# Iterate through the image 
	# 			for x in range(0, WIDTH - bW*j + 1):
	# 				features[count] = -intimg[y,x] + intimg[y,x+bW*j] \
	# 								+ 2*intimg[y+bH*i/2,x] - 2*intimg[y+bH*i/2,x+bW*j] \
	# 								- intimg[y+bH*i,x] + intimg[y+bH*i,x+bW*j]
	# 				count = count + 1
	# print "Total number of features...", count
	# return features

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
	# Vertical features, 2x1 base size
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

def _extract_features(data, feature_matrix):
	'''
		Extract all features from integral images into a nFeatures x nSamples array.
		Also return the sorted indices along the samples direction.
	'''
	# We are only concerned with two-rectangle edge features
	# Get the sorted list of sample indices.
	feature_vectors = dot( feature_matrix, data )
	sorted_indices = argsort(feature_vectors, axis=1)
	return sorted_indices, feature_vectors

class CascadedAdaBoostClassifier(object):
	def __init__(self):
		super(CascadedAdaBoostClassifier, self).__init__()
		self.cascaded_classfiers = []
		self.train_data = None
		self.train_label = None
		self.train_sorted_indices = None
		self.train_feat_vecs = None
		self.train_num_pos = 0
		self.train_num_neg = 0
		self.test_data = None
		self.test_label = None
		self.test_sorted_indices = None
		self.test_feat_vecs = None
		self.test_num_pos = 0
		self.test_num_neg = 0
		self.feature_matrix = _get_feature_matrix()

	def set_training_data(train_data, train_label):
		self.train_sorted_indices, self.train_feat_vecs = _extract_features(train_data, self.feature_matrix)
		self.train_num_pos = sum(train_label)
		self.train_num_neg = train_label.size - self.train_num_pos
		self.train_data, self.train_label = train_data, train_label

	def set_testing_data(test_data, test_label):
		self.test_sorted_indices, self.test_feat_vecs = _extract_features(test_data, self.feature_matrix)
		self.test_num_pos = sum(test_label)
		self.test_num_neg = test_label.size - self.test_num_pos
		self.test_data, self.test_label = test_data, test_label

	def train(self, f, d, Ftarg, maxIter):
		'''
			Train cascaded AdaBoost classifiers given user-defined:
			max acceptable fpr per layer f, min acceptable detection rate per layer d,
			and overall fpr F_target.
		'''
		i = 1
		# Preprocess the features
		Fprev, Dprev = 1.0, 1.0
		Fcurr = 1.0
		# Every iteration adds a new AdaBoost classifier
		while Fcurr > Ftarg and i < maxIter:
			i = i + 1
			Fcurr = Fprev
			# Add another AdaBoost classifier to our cascade
			current_adaboost = _add_adaboost_classifier()
			# Every iteration adds a new feature
			while Fcurr > f * Fprev:
				# Add a new weak classifier to our current Adaboost classifier
				current_adaboost.add_weak_classifier()
				# Evaluate the current cascade
				Fcurr, Dcurr = self.test()
				# Decrease threshold until current cascade exceeds a certain detection rate
				while Dcurr < d * Dprev:
					current_adaboost.decrease_threshold(0.02)
					Fcurr, Dcurr = self.test()

	def _add_adaboost_classifier():
		'''
			Allocate and return a new AdaBoost classifier.
		'''
		c = AdaBoostClassifier()
		c.set_feature_matrix(self.feature_matrix)
		c.set_training_data(self.train_sorted_indices, self.train_feat_vecs, self.train_label)
		c.set_testing_data(self.test_sorted_indices, self.test_feat_vecs, self.train_label):
		self.cascaded_classfiers.append[c]
		return c

	def test():
		'''
			Evaluate the cascade on test data and return FPR and detection rate.
		'''
		# Evaluate all the test samples in the beginning
		positive_indices = arange(test_label.size)
		for classifier in self.cascaded_classfiers:
			positive_indices = classifier.test(positive_indices)
		# Calculate detection rate by counting the number of ones in the ground-truth of the predicted true samples
		num_true_positive = sum(test_label[positive_indices])
		D = num_true_positive*1.0 / self.test_num_pos
		# Calculate false positive rate by counting the zeros
		F = (positive_indices.size - num_true_positive)*1.0 / self.test_num_neg
		return F, D

class AdaBoostClassifier(object):
	def __init__(self):
		super(AdaBoostClassifier, self).__init__()
		self.train_data = None
		self.train_label = None
		self.train_sorted_indices = None
		self.train_feat_vecs = None
		self.train_num_pos = 0
		self.train_num_neg = 0
		self.test_data = None
		self.test_label = None
		self.test_sorted_indices = None
		self.test_feat_vecs = None
		self.test_num_pos = 0
		self.test_num_neg = 0
		self.threshold = 1.0
		self.weights = None

	def set_feature_matrix(feature_matrix):
		self.feature_matrix = feature_matrix

	def set_training_data(train_sorted_indices, train_feat_vecs, train_label):
		self.train_sorted_indices, self.train_feat_vecs = train_sorted_indices, train_feat_vecs
		self.train_num_pos = sum(train_label)
		self.train_num_neg = train_label.size - self.train_num_pos
		self.train_label = train_label

	def set_testing_data(test_sorted_indices, test_feat_vecs, test_label):
		self.test_sorted_indices, self.test_feat_vecs = test_sorted_indices, test_feat_vecs
		self.test_num_pos = sum(test_label)
		self.test_num_neg = test_label.size - self.test_num_pos
		self.test_label = test_label

	def add_weak_classifier():
		'''
			Add the current best weak classifier on the weighted training set.
		'''
		# Initialize all the weights if this is the first weak classifier
		if not self.weights:
			self.weights = array(self.test_label.size, dtype=float)
			self.weights.fill( 1.0 / (2 * self.train_num_neg) )
			self.weights[self.train_label].fill( 1.0 / (2 * self.train_num_pos) )
		# Normalize the weights otherwise
		else:
			self.weights = self.weights / sum(self.weights)

	def _get_best_weak_classifier():
		'''
			Return the index of the best feature with the minimum weighted error.
		'''
		

	def decrease_threshold(step):
		'''
			Decrease the classification threshold by step.
		'''
		self.threshold = self.threshold - step
		assert self.threshold >= 0.0, "AdaBoost decision threshold cannot be negative!"

	def test(test_indices):
		'''
			Evaluate the strong classifier's performance given a decision threshold on given samples within the test data.
			Return only the indices of the positive samples.
		'''
		pass