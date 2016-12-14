#!/usr/bin/python
from pylab import *
import cv2

WIDTH = 40
HEIGHT = 20
# NUMFEATURES = 10 # Obtained by _get_feature_matrix()
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
	print "Constructing feature matrix...", features.shape
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
					# if count == NUMFEATURES: return features
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
	print "Feature vectors size...", feature_vectors.shape
	print "Sorting samples based on feature values..."
	sorted_indices = argsort(feature_vectors, axis=1)
	print "Sorted indices size...", sorted_indices.shape
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

	def set_training_data(self, train_data, train_label):
		self.train_sorted_indices, self.train_feat_vecs = _extract_features(train_data, self.feature_matrix)
		self.train_num_pos = sum(train_label)
		self.train_num_neg = train_label.size - self.train_num_pos
		self.train_data, self.train_label = train_data, train_label

	def set_testing_data(self, test_data, test_label):
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
		Flog = []
		Dlog = []
		Fcurr = 1.0
		# Every iteration adds a new AdaBoost classifier
		while Fcurr > Ftarg and i < maxIter:
			print "Training %dth AdaBoost classifier in the cascade..." % i
			i = i + 1
			Fcurr = Fprev
			# Add another AdaBoost classifier to our cascade
			current_adaboost = self._add_adaboost_classifier()
			# Every iteration adds a new feature
			# while Fcurr > f * Fprev:
			for j in range(40):
				print "Adding a new feature..."
				# Add a new weak classifier to our current Adaboost classifier
				current_adaboost.add_weak_classifier()
				# Evaluate the current cascade
				Fcurr, Dcurr = self.test()
				# Decrease threshold until current cascade exceeds a certain detection rate
				# while Dcurr < d * Dprev:
				# 	current_adaboost.decrease_threshold(0.02)
				# 	Fcurr, Dcurr = self.test()
			Fprev = Fcurr
			Dprev = Dcurr
			Flog.append(Fcurr)
			Dlog.append(Dcurr)
		print "FP: ", Flog
		print "RC: ", Dlog

	def _add_adaboost_classifier(self):
		'''
			Allocate and return a new AdaBoost classifier.
		'''
		c = AdaBoostClassifier()
		c.set_feature_matrix(self.feature_matrix)
		c.set_training_data(self.train_sorted_indices, self.train_feat_vecs, self.train_label)
		c.set_testing_data(self.test_sorted_indices, self.test_feat_vecs, self.train_label)
		self.cascaded_classfiers.append(c)
		return c

	def test(self):
		'''
			Evaluate the cascade on test data and return FPR and detection rate.
		'''
		# Evaluate all the test samples in the beginning
		positive_indices = arange(self.train_label.size)
		# print self.test_label
		for classifier in self.cascaded_classfiers:
			positive_indices = classifier.test(positive_indices)
		# Calculate detection rate by counting the number of ones in the ground-truth of the predicted true samples
		num_true_positive = sum(self.train_label[positive_indices])
		D = num_true_positive*1.0 / self.train_num_pos
		# Calculate false positive rate by counting the zeros
		F = (positive_indices.size - num_true_positive)*1.0 / self.train_num_pos
		print "F = %.4f, D = %.4f" % (F,D)
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
		self.weak_classifier_indices = array([], dtype=int)
		self.weak_classifier_polarities = array([])
		self.weak_classifier_threshs = array([])
		self.weak_classifier_weights = array([])
		self.weak_classifier_results = array([])
		self.weak_classifier_weighted_results = None

	def set_feature_matrix(self, feature_matrix):
		self.feature_matrix = feature_matrix

	def set_training_data(self, train_sorted_indices, train_feat_vecs, train_label):
		self.train_sorted_indices, self.train_feat_vecs = train_sorted_indices, train_feat_vecs
		self.train_num_pos = int(sum(train_label))
		self.train_num_neg = train_label.size - self.train_num_pos
		self.train_label = train_label
		print "Number of positive / negative samples in training...", self.train_num_pos, self.train_num_neg

	def set_testing_data(self, test_sorted_indices, test_feat_vecs, test_label):
		self.test_sorted_indices, self.test_feat_vecs = test_sorted_indices, test_feat_vecs
		self.test_num_pos = int(sum(test_label))
		self.test_num_neg = test_label.size - self.test_num_pos
		self.test_label = test_label
		print "Number of positive / negative samples in testing...", self.test_num_pos, self.test_num_neg

	def add_weak_classifier(self):
		'''
			Add the current best weak classifier on the weighted training set.
		'''
		# Initialize all the weights if this is the first weak classifier
		if self.weak_classifier_indices.size == 0:
			self.weights = zeros(self.train_label.size, dtype=float)
			self.weights.fill( 1.0 / (2 * self.train_num_neg) )
			self.weights[self.train_label==1].fill( 1.0 / (2 * self.train_num_pos) )
		# Normalize the weights otherwise
		else:
			self.weights = self.weights / sum(self.weights)
		# Now pick the weak classifier with the min error with respect to the current weights
		best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error, best_feat_results = self._get_best_weak_classifier()
		# Update our list of weak classifiers
		self.weak_classifier_indices = append(self.weak_classifier_indices, best_feat_index)
		self.weak_classifier_polarities = append(self.weak_classifier_polarities, best_feat_polarity)
		self.weak_classifier_threshs = append(self.weak_classifier_threshs, best_feat_thresh)
		# Get confidence value of the best new classifier
		# Following the notation in the paper
		beta = best_feat_error / (1 - best_feat_error)
		alpha = log(1 / beta)
		self.weak_classifier_weights = append(self.weak_classifier_weights, alpha)
		# print "New best weak classifier"
		# print "Indices, Polarities, Threshs, Weights"
		# print best_feat_index
		# print best_feat_polarity
		# print best_feat_thresh
		# print alpha
		# print "Corresponding feature row"
		# print self.train_feat_vecs[best_feat_index, self.train_sorted_indices[best_feat_index, :]]
		# print ""
		# Update the weights of the samples
		# classified_labels = zeros(self.train_num_pos + self.train_num_neg)
		# classified_labels[ best_feat_polarity*self.train_feat_vecs[best_feat_index, :] > best_feat_polarity*best_feat_thresh ] = 1
		# e = abs(classified_labels - self.train_label)
		# print "best_feat_results", best_feat_results
		e = abs(best_feat_results - self.train_label)
		self.weights = self.weights * beta**(1-e)
		# Adjust the threshold
		if self.weak_classifier_results.size == 0:
			self.weak_classifier_results = best_feat_results.reshape(-1,1)
		else:
			self.weak_classifier_results = hstack((self.weak_classifier_results, best_feat_results.reshape(-1,1)))
		self.weak_classifier_weighted_results = dot(self.weak_classifier_results, self.weak_classifier_weights)
		print self.weak_classifier_results
		print self.weak_classifier_weighted_results
		self.threshold = sum(self.weak_classifier_weights)/2
		# self.threshold = min(self.weak_classifier_weighted_results[self.train_label==1])
		print "Threshold =", self.threshold

	def _get_best_weak_classifier(self):
		'''
			Return the index of the best feature with the minimum weighted error.
		'''
		feature_errors = zeros(NUMFEATURES)
		feature_thresh = zeros(NUMFEATURES)
		feature_polarity = zeros(NUMFEATURES)
		feature_sorted_index = zeros(NUMFEATURES)
		Tplus = sum(self.weights[self.train_label==1])
		Tminus = sum(self.weights[self.train_label==0])
		# For loop sucks
		for r in range(NUMFEATURES):
			sorted_weights = self.weights[self.train_sorted_indices[r,:]]
			sorted_labels = self.train_label[self.train_sorted_indices[r,:]]
			# print "===="
			# print "sorted_weights", sorted_weights
			# print "sorted_labels", sorted_labels
			Splus = cumsum(sorted_labels * sorted_weights)
			Sminus = cumsum((1-sorted_labels) * sorted_weights)
			# print "Splus", Splus
			# print "Sminus", Sminus
			# Error of choice influences the polarity
			polarities = ones(self.train_num_pos + self.train_num_neg)
			polarities[Splus + Tminus - Sminus > Sminus + Tplus - Splus] = -1
			# print polarities
			errors = minimum(Splus + Tminus - Sminus, Sminus + Tplus - Splus)
			# print "errors", errors
			sorted_index = argmin(errors)
			min_error_sample_index = self.train_sorted_indices[r,sorted_index]
			# print "min_error_sample_index", min_error_sample_index
			min_error = min(errors)
			# print "min_error", min_error
			threshold = self.train_feat_vecs[r, min_error_sample_index]
			polarity = polarities[min_error_sample_index]
			# print "threshold", threshold
			feature_errors[r] = min_error
			feature_thresh[r] = threshold
			feature_polarity[r] = polarity
			feature_sorted_index[r] = sorted_index
		# Now pick the best one
		best_feat_index = argmin(feature_errors)
		best_feat_thresh = feature_thresh[best_feat_index]
		best_feat_error = feature_errors[best_feat_index]
		best_feat_polarity = feature_polarity[best_feat_index]
		best_feat_results = zeros(self.train_num_pos + self.train_num_neg)
		best_sorted_index = feature_sorted_index[best_feat_index]
		if best_feat_polarity == 1:
			best_feat_results[ self.train_sorted_indices[best_feat_index, best_sorted_index:] ] = 1
		else:
			best_feat_results[ self.train_sorted_indices[best_feat_index, :best_sorted_index] ] = 1
		# print best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error
		return best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error, best_feat_results

	def test(self, selected_indices):
		'''
			Evaluate the strong classifier's performance given a decision threshold on given samples within the test data.
			Return only the indices of the positive samples.
		'''
		# # Following notations in the paper
		# sum_alpha = sum(self.weak_classifier_weights)
		# selected_feat_vecs = self.test_feat_vecs[ :, selected_indices ]
		# # Only preserve the feature vector where the feature is used
		# # print self.weak_classifier_indices
		# selected_feat_vecs = selected_feat_vecs[self.weak_classifier_indices, :]
		# polarized_threshs = self.weak_classifier_polarities * self.weak_classifier_threshs
		# h = zeros(selected_feat_vecs.shape)
		# h[ dot(self.weak_classifier_polarities, selected_feat_vecs) > polarized_threshs.reshape(-1,1)] = 1.0
		# # print h
		# # h_polarized = abs(self.weak_classifier_polarities - h)
		# pred = zeros(selected_indices.size)
		# pred[ dot(self.weak_classifier_weights, h) > sum_alpha * self.threshold ] = 1
		# # print pred
		positive_selected_indices = self.weak_classifier_weighted_results[selected_indices] > self.threshold
		return selected_indices[ positive_selected_indices ]
