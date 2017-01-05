#!/usr/bin/python
from pylab import *
import cv2

WIDTH = 40
HEIGHT = 20
NUMFEATURES = 47232 # Obtained by _get_feature_matrix()

def get_integral_image(image):
	'''
		Compute and return the integral representation of an gray-scale image.
	'''
	return cumsum(cumsum(image,axis=0),axis=1)

def _get_feature_matrix():
	'''
		Return the feature matrix.
	'''
	features = zeros((NUMFEATURES, WIDTH*HEIGHT), dtype=int8)
	print "Constructing feature matrix...", features.shape
	count = 0
	# Offset from image boundary
	offset = 2
	# Do horizontal ones first, 1x2 base size
	bW, bH = 2, 1
	for i in range(1, HEIGHT, bH): 					# Extend row-wise multiplier
		for j in range(1, WIDTH, bW): 				# Extend column-wise multiplier
			for y in range(0 + offset, HEIGHT - bH*i + 1 - offset):	# Iterate through the image 
				for x in range(0 + offset, WIDTH - bW*j + 1 - offset):
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
			for y in range(0 + offset, HEIGHT - bH*i + 1 - offset):	# Iterate through the image 
				for x in range(0 + offset, WIDTH - bW*j + 1 - offset):
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
		self.cascaded_adaboost = []
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

	def set_testing_data(self, test_data, test_label):
		self.test_data = test_data
		self.test_label = test_label

	def train(self, train_data, train_label, num_stages, num_feats):
	# def train(self, train_data, train_label, f, d, Ftarg, maxIter):
		'''
			Train cascaded AdaBoost classifiers given user-defined:
			max acceptable fpr per layer f, min acceptable detection rate per layer d,
			and overall fpr F_target.
		'''
		self.train_feat_vecs = dot( self.feature_matrix, train_data )
		self.train_num_pos = int(sum(train_label))
		self.train_num_neg = train_label.size - self.train_num_pos
		self.train_num = train_label.size
		self.train_data, self.train_label = train_data, train_label

		# Postive and negative training samples for current stage
		all_pos_feat_vecs = self.train_feat_vecs[:,self.train_label==1]
		all_neg_feat_vecs = self.train_feat_vecs[:,self.train_label==0]
		pos_feat_vecs = all_pos_feat_vecs
		neg_feat_vecs = all_neg_feat_vecs

		Flog_train = []
		Dlog_train = []
		Alog_train = []
		Flog_test = []
		Dlog_test = []
		Alog_test = []

		# Add stages
		for i in range(num_stages): 
			print "Training %dth AdaBoost classifier in the cascade..." % i+1
			current_adaboost = self._add_adaboost_classifier()
			current_adaboost.set_training_feature_vectors(pos_feat_vecs, neg_feat_vecs)
			# Add features
			for j in range(num_feats):
				print "Adding feature %d..." % j+1
				current_adaboost.add_weak_classifier()
			# Update negative samples to use
			fp_indices,F,D,A = self._classify_training_data()

			Flog_train.append(F)	
			Dlog_train.append(D)
			Alog_train.append(A)

			neg_feat_vecs = all_neg_feat_vecs[:,fp_indices-self.train_num_pos]

			F,D,A = self.test()

			Flog_test.append(F)	
			Dlog_test.append(D)
			Alog_test.append(A)

			print "Training:"
			print "FP:\n", Flog_train
			print "RC:\n", Dlog_train
			print "AC:\n", Alog_train
			print "Testing:"
			print "FP:\n", Flog_test
			print "RC:\n", Dlog_test
			print "AC:\n", Alog_test

	def _add_adaboost_classifier(self):
		'''
			Allocate and return a new AdaBoost classifier.
		'''
		c = AdaBoostClassifier()
		c.set_feature_matrix(self.feature_matrix)
		self.cascaded_adaboost.append(c)
		return c

	def _classify_training_data(self):
		'''
			Evaluate the cascaded classifier on the training data,
			and return the indices of false postive samples as well as the rates.
		'''
		print "Classifying training images..."
		feat_vecs = self.train_feat_vecs
		pos_indices = arange(self.train_num)
		for classifier in self.cascaded_adaboost:
			preds = classifier.classify_feature_vectors(feat_vecs)
			# Only pass on the samples with postive predictions
			feat_vecs = feat_vecs[:,preds==1]
			pos_indices = pos_indices[preds==1]
		# Final prediction
		fp_indices = pos_indices[ self.train_label[pos_indices] == 0 ]
		num_tp = sum(self.train_label[pos_indices])
		D = num_tp*1.0 / self.train_num_pos
		# Calculate false positive rate by counting the zeros
		F = (pos_indices.size - num_tp)*1.0 / self.train_num_neg
		w = self.train_num_pos*1.0 / (self.train_num_pos + self.train_num_neg)
		A = D * w + (1-F)*(1-w)
		print "F = %.4f, D = %.4f, A = %.4f" % (F,D,A)
		return fp_indices, F, D, A

	def test(self):
		'''
			Classify test images.
		'''
		print "Classifying testing images..."
		feat_vecs = dot( self.feature_matrix, self.test_data )
		test_num_pos = int(sum(self.test_label))
		test_num_neg = self.test_label.size - test_num_pos
		test_num = self.test_label.size
		pos_indices = arange(test_num)
		for classifier in self.cascaded_adaboost:
			preds = classifier.classify_feature_vectors(feat_vecs)
			# Only classify the samples with postive predictions
			feat_vecs = feat_vecs[:,preds==1]
			pos_indices = pos_indices[preds==1]
		# Final prediction
		num_tp = sum(self.test_label[pos_indices])
		D = num_tp*1.0 / test_num_pos
		# Calculate false positive rate by counting the zeros
		F = (pos_indices.size - num_tp)*1.0 / test_num_neg
		w = test_num_pos*1.0 / (test_num_pos + test_num_neg)
		A = D * w + (1-F)*(1-w)
		print "F = %.4f, D = %.4f, A = %.4f" % (F,D,A)
		return F,D,A

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
		self.sample_weights = None
		self.weak_classifier_indices = array([], dtype=int)
		self.weak_classifier_polarities = array([])
		self.weak_classifier_threshs = array([])
		self.weak_classifier_weights = array([])
		self.weak_classifier_results = array([])
		self.weak_classifier_weighted_results = None

	def set_feature_matrix(self, feature_matrix):
		self.feature_matrix = feature_matrix

	def set_training_feature_vectors(self, pos_feat_vecs, neg_feat_vecs):
		'''
			Given current training feature vectors, sort them.
		'''
		self.train_num_pos = pos_feat_vecs.shape[1]
		self.train_num_neg = neg_feat_vecs.shape[1]
		self.train_label = hstack( (ones(self.train_num_pos), zeros(self.train_num_neg)) ) 
		self.train_feat_vecs = hstack( (pos_feat_vecs, neg_feat_vecs) )
		self.train_sorted_indices = argsort(self.train_feat_vecs, axis=1)
		print "Number of positive / negative samples in training...", self.train_num_pos, self.train_num_neg

	def add_weak_classifier(self):
		'''
			Add the current best weak classifier on the weighted training set.
		'''
		# Initialize all the weights if this is the first weak classifier
		if self.weak_classifier_indices.size == 0:
			self.sample_weights = zeros(self.train_label.size, dtype=float)
			self.sample_weights.fill( 1.0 / (2 * self.train_num_neg) )
			self.sample_weights[self.train_label==1] = 1.0 / (2 * self.train_num_pos) 
		# Normalize the weights otherwise
		else:
			self.sample_weights = self.sample_weights / sum(self.sample_weights)
		# Now pick the weak classifier with the min error with respect to the current weights
		best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error, best_feat_results = self._get_best_weak_classifier()
		# Update our list of weak classifiers
		self.weak_classifier_indices = append(self.weak_classifier_indices, best_feat_index)
		self.weak_classifier_polarities = append(self.weak_classifier_polarities, best_feat_polarity)
		self.weak_classifier_threshs = append(self.weak_classifier_threshs, best_feat_thresh)
		# Get confidence value of the best new classifier
		# Following the notation in the paper
		beta = best_feat_error / (1 - best_feat_error)
		alpha = log(1 / abs(beta))
		self.weak_classifier_weights = append(self.weak_classifier_weights, alpha)
		e = abs(best_feat_results - self.train_label)
		self.sample_weights = self.sample_weights * beta**(1-e)
		# Adjust the threshold
		if self.weak_classifier_results.size == 0:
			self.weak_classifier_results = best_feat_results.reshape(-1,1)
		else:
			self.weak_classifier_results = hstack((self.weak_classifier_results, best_feat_results.reshape(-1,1)))
		self.weak_classifier_weighted_results = dot(self.weak_classifier_results, self.weak_classifier_weights)
		# print self.weak_classifier_results,"weak_classifier_results"
		# print self.weak_classifier_weights, "weak_classifier_weights"
		# print self.weak_classifier_weighted_results, "weak_classifier_weighted_results"
		# self.threshold = sum(self.weak_classifier_weights)/2
		self.threshold = min(self.weak_classifier_weighted_results[self.train_label==1])

	def _get_best_weak_classifier(self):
		'''
			Return the index of the best feature with the minimum weighted error.
		'''
		feature_errors = zeros(NUMFEATURES)
		feature_thresh = zeros(NUMFEATURES)
		feature_polarity = zeros(NUMFEATURES)
		feature_sorted_index = zeros(NUMFEATURES, dtype=int)
		Tplus = sum(self.sample_weights[self.train_label==1])
		Tminus = sum(self.sample_weights[self.train_label==0])
		# Iterate to find the best feature
		for r in range(NUMFEATURES):
			sorted_weights = self.sample_weights[self.train_sorted_indices[r,:]]
			sorted_labels = self.train_label[self.train_sorted_indices[r,:]]
			Splus = cumsum(sorted_labels * sorted_weights)
			Sminus = cumsum(sorted_weights) - Splus
			# Error of choice influences the polarity
			Eplus = Splus + Tminus - Sminus
			Eminus = Sminus + Tplus - Splus
			polarities = zeros(self.train_num_pos + self.train_num_neg)
			polarities[Eplus > Eminus] = -1
			polarities[Eplus <= Eminus] = 1
			errors = minimum(Eplus, Eminus)
			sorted_index = argmin(errors)
			min_error_sample_index = self.train_sorted_indices[r,sorted_index]
			min_error = min(errors)
			threshold = self.train_feat_vecs[r, min_error_sample_index]
			polarity = polarities[sorted_index]
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
		print 'best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error'
		print best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error
		return best_feat_index, best_feat_polarity, best_feat_thresh, best_feat_error, best_feat_results

	def classify_feature_vectors(self, feat_vecs):
		'''
			Classify feature vectors and return classified labels.		
		'''
		# Get the feature values
		weak_classifiers = feat_vecs[self.weak_classifier_indices,:]
		# Organize as column vectors to ease broadcasting later
		polar_colvec = self.weak_classifier_polarities.reshape(-1,1)
		thresh_colvec = self.weak_classifier_threshs.reshape(-1,1)
		# Predictions of all weak classifiers
		weak_classifier_preds = weak_classifiers * polar_colvec > thresh_colvec * polar_colvec
		weak_classifier_preds[weak_classifier_preds==True] = 1
		weak_classifier_preds[weak_classifier_preds==False] = 0
		# Apply weak classifier weights
		strong_classifier_result = dot(self.weak_classifier_weights, weak_classifier_preds)
		# Apply strong classifier threshold
		final_preds = zeros(strong_classifier_result.size)
		final_preds[strong_classifier_result >= self.threshold] = 1
		return final_preds