#!/usr/bin/python
from pylab import *
import cv2
import os
from pca import PCAClassifier
from lda import LDAClassifier
from adaboost import *

def load_face_dataset():
	'''
		Load the face dataset in the following format:
		- each image is converted to gray scale
		- each face image is vectorized as a column vector
		- labels are organized as a column vector
	'''
	print "Loading face dataset..."
	# Process training images first
	train_path = './face-dataset/train/'
	train_files = [f for f in os.listdir(train_path) if f.endswith(".png")]
	num_train = len(train_files)
	train_data = zeros((128*128, num_train), dtype=float)
	train_label = zeros(num_train, dtype=int)
	for i,f in enumerate(train_files):
		image = imread(os.path.join(train_path,f))
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		train_data[:,i] = image.flatten()
		train_label[i] = int(f.split('_')[0])
	# Normalization across all images, zero mean and unit variance
	train_mean = mean(train_data)
	train_std = std(train_data)
	train_data = train_data - train_mean
	train_data = train_data / train_std
	# Process testing images
	test_path = './face-dataset/test/'
	test_files = [f for f in os.listdir(test_path) if f.endswith(".png")]
	num_test = len(test_files)
	test_data = zeros((128*128, num_test), dtype=float)
	test_label = zeros(num_test, dtype=int)
	for i,f in enumerate(test_files):
		image = imread(os.path.join(test_path,f))
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		test_data[:,i] = image.flatten()
		test_label[i] = int(f.split('_')[0])
	# Normalization using training information
	test_data = test_data - train_mean
	test_data = test_data / train_std
	print "Loading finished..."
	print "Sizes...", train_data.shape, train_label.shape, test_data.shape, test_label.shape
	return train_data, train_label, test_data, test_label

def load_car_dataset():
	'''
		Load the car dataset in the following format:
		- each image is converted to gray scale
		- each car image is vectorized as a column vector
		- labels are organized as a column vector
	'''
	print "Loading car dataset..."
	# Process training images first
	train_path = './car-dataset/train/'
	train_pos_files = [f for f in os.listdir(os.path.join(train_path, 'positive'))]
	train_neg_files = [f for f in os.listdir(os.path.join(train_path, 'negative'))]
	num_train = len(train_pos_files + train_neg_files)
	train_pos_data = zeros((40*20, len(train_pos_files)))
	train_neg_data = zeros((40*20, len(train_neg_files)))
	for i,f in enumerate(train_pos_files):
		image = imread(os.path.join(train_path, 'positive', f))
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		image = get_integral_image(image)
		train_pos_data[:,i] = image.flatten()
	for i,f in enumerate(train_neg_files):
		image = imread(os.path.join(train_path, 'negative', f))
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		image = get_integral_image(image)
		train_neg_data[:,i] = image.flatten()
	# Process testing images
	test_path = './car-dataset/test/'
	test_pos_files = [f for f in os.listdir(os.path.join(test_path, 'positive'))]
	test_neg_files = [f for f in os.listdir(os.path.join(test_path, 'negative'))]
	num_test = len(test_pos_files + test_neg_files)
	test_pos_data = zeros((40*20, len(test_pos_files)))
	test_neg_data = zeros((40*20, len(test_neg_files)))
	for i,f in enumerate(test_pos_files):
		image = imread(os.path.join(test_path, 'positive', f))
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		image = get_integral_image(image)
		test_pos_data[:,i] = image.flatten()
	for i,f in enumerate(test_neg_files):
		image = imread(os.path.join(test_path, 'negative', f))
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		image = get_integral_image(image)
		test_neg_data[:,i] = image.flatten()
	train_data = hstack((train_pos_data, train_neg_data))
	train_label = hstack(( ones(train_pos_data.shape[1]), zeros(train_neg_data.shape[1]) ))
	test_data = hstack((test_pos_data, test_neg_data))
	test_label = hstack(( ones(test_pos_data.shape[1]), zeros(test_neg_data.shape[1]) ))
	print "Loading finished..."
	print "Sizes...", train_data.shape, train_label.shape, test_data.shape, test_label.shape
	print "Type...", train_data.dtype, train_label.dtype
	return train_data, train_label, test_data, test_label

def PCVvsLDA(max_K):
	train_data, train_label, test_data, test_label = load_face_dataset()
	pca_accu = zeros(max_K)
	lda_accu = zeros(max_K)
	for k in range(max_K):
		pca = PCAClassifier()
		pca.train(train_data, train_label, k+1)
		pca_accu[k] = pca.test(test_data, test_label)
		lda = LDAClassifier()
		lda.train(train_data, train_label, k+1)
		lda_accu[k] = lda.test(test_data, test_label)
	line1, = plot(linspace(1,max_K,num=max_K), pca_accu, '-ro', label='PCA')
	line2, = plot(linspace(1,max_K,num=max_K), lda_accu, '-go', label='LDA')
	legend(handles=[line1, line2], loc=4)
	xlabel('K')
	ylabel('Accuracy')
	show()	

def main():
	algorithms = ['AdaBoost'] # 'PCA', 'LDA', 'AdaBoost'
	# PCA
	if 'PCA' in algorithms:
		train_data, train_label, test_data, test_label = load_face_dataset()
		pca = PCAClassifier()
		pca.train(train_data, train_label, 60)
		pca.test(test_data, test_label)
	# LDA
	elif 'LDA' in algorithms:
		train_data, train_label, test_data, test_label = load_face_dataset()
		lda = LDAClassifier()
		lda.train(train_data, train_label, 30)
		lda.test(test_data, test_label)
	elif 'AdaBoost' in algorithms:
		f = 0.3
		d = 0.9
		T = 10
		S = 10
		Ftarg = 0.01
		maxIter = 3
		train_data, train_label, test_data, test_label = load_car_dataset()
		violajones = CascadedAdaBoostClassifier()
		# violajones.set_training_data(train_data, train_label)
		violajones.set_testing_data(test_data, test_label)
		violajones.train(train_data, train_label)

if __name__ == '__main__':
	main()
