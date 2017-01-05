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

def PCAvsLDA(max_K):
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

def plot_adaboost():
	F_train = array([0.3049, 0.2088, 0.1456, 0.1303, 0.1200, 0.1041, 0.1018, 0.0910, 0.0899])*100.0
	D_train = array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9986, 0.9972])*100.0
	A_train = array([0.7828, 0.8513, 0.8963, 0.9072, 0.9145, 0.9259, 0.9275, 0.9348, 0.9352])*100.0
	F_test = array([0.3341, 0.2477, 0.1773, 0.1568, 0.1432, 0.1409, 0.1386, 0.1318, 0.1318])*100.0
	D_test = array([0.9831, 0.9663, 0.9551, 0.9438, 0.9438, 0.9382, 0.9382, 0.9382, 0.9382])*100.0
	A_test = array([0.7573, 0.8139, 0.8608, 0.8722, 0.8819, 0.8819, 0.8835, 0.8883, 0.8883])*100.0
	N = len(F_train)
	for train, test, name, loc in [(F_train, F_test, 'False Positive Rate', 1),
								(D_train, D_test, 'Detection Rate', 3),
								(A_train, A_test, 'Accuracy', 4)]:
		figure()
		line1, = plot(range(1,N+1), train, '-ro', label='Train')
		line2, = plot(range(1,N+1), test, '-go', label='Test')
		legend(handles=[line1, line2], loc=loc)
		xlabel('N')
		ylabel(name + ' (%)')
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
		num_stages = 10
		num_feats = 20
		train_data, train_label, test_data, test_label = load_car_dataset()
		violajones = CascadedAdaBoostClassifier()
		violajones.set_testing_data(test_data, test_label)
		violajones.train(train_data, train_label, num_stages, num_feats)

if __name__ == '__main__':
	main()
