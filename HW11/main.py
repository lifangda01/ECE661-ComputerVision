#!/usr/bin/python
from pylab import *
import cv2
import os
from pca import PCAClassifier
from lda import LDAClassifier

def load_dataset():
	'''
		Load the face dataset in the following format:
		- each image is converted to gray scale
		- each face image is vectorized as a column vector
		- labels are organized as a column vector
	'''
	print "Dataset loading..."
	# Process training images first
	train_path = './dataset/train/'
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
	test_path = './dataset/test/'
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
	print "Dataset loading finished..."
	print "Sizes...", train_data.shape, train_label.shape, test_data.shape, test_label.shape
	return train_data, train_label, test_data, test_label

def main():
	algorithms = ['LDA']
	train_data, train_label, test_data, test_label = load_dataset()
	# PCA
	if 'PCA' in algorithms:
		pca = PCAClassifier(train_data, train_label, 60)
		pca.train()
		pca.test(test_data, test_label)
	# LDA
	elif 'LDA' in algorithms:
		lda = LDAClassifier(train_data, train_label, 30)
		lda.train()
		lda.test(test_data, test_label)
	elif 'AdaBoost' in algorithms:
		pass

if __name__ == '__main__':
	main()