#!/usr/bin/python
import os
from pylab import *
import cv2
from sklearn.neighbors import NearestNeighbors
from lbp import get_lbp_hist
from mpl_toolkits.mplot3d import Axes3D

def train(R, P):
	'''
		Extract LBP features from training images.
		@R: int of number of LBP sampling circle radius
		@P: int of number of samples to take on each circle
	'''
	tdir = os.path.join(os.getcwd(),'imagesDatabaseHW8','training')
	classes = os.listdir(tdir)
	hists = None
	labels = None
	print "Training started..."
	for c in classes:
		sdir = os.path.join(tdir,c)
		samples = os.listdir(sdir)
		label = classes.index(c)
		for s in samples:
			# Read the image and convert to gray scale
			fpath = os.path.join(sdir,s)
			image = imread(fpath)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			# Get the LBP histogram
			hist = get_lbp_hist(image, R, P)
			if hists != None:
				hists = vstack( (hists, hist) )
				labels = append(labels, label)
			else:
				hists = hist
				labels = array([label])
			print "Training finished on...", fpath
	savetxt("train_hists.out", hists)
	savetxt("train_labels.out", labels)
	print "Trained histogram and labels saved..."

def test(R, P):
	'''
		Extract LBP features from testing images.
		@R: int of number of LBP sampling circle radius
		@P: int of number of samples to take on each circle
	'''
	tdir = os.path.join(os.getcwd(),'imagesDatabaseHW8','training')
	classes = os.listdir(tdir)
	tdir = os.path.join(os.getcwd(),'imagesDatabaseHW8','testing')
	samples = os.listdir(tdir)
	hists = None
	labels = None
	print "Testing started..."
	for s in samples:
		# Read the image and convert to gray scale
		fpath = os.path.join(tdir,s)
		image = imread(fpath)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		# Obtain ground truth index
		label = classes.index( s.split("_")[0] )
		# Get the LBP histogram
		hist = get_lbp_hist(image, R, P)
		if hists != None:
			hists = vstack( (hists, hist) )
			labels = append(labels, label)
		else:
			hists = hist
			labels = array([label])
		print "Testing finished on...", fpath
	savetxt("test_hists.out", hists)
	savetxt("test_labels.out", labels)
	print "Testing histogram and labels saved..."

def evaluate(n, weightedNN=True):
	'''
		Evaluate our classifier.
		@n: int of number of nearest neighbors for majority voting
	'''
	trainHists = loadtxt("train_hists.out")
	trainLabels = loadtxt("train_labels.out").astype(uint8)
	testHists = loadtxt("test_hists.out")
	testLabels = loadtxt("test_labels.out").astype(uint8)
	nSamples = testLabels.size
	# Construct NN
	# Each sample hist is a row vector
	NN = NearestNeighbors(n_neighbors=n, metric='euclidean').fit(trainHists)
	# Get classes information
	tdir = os.path.join(os.getcwd(),'imagesDatabaseHW8','training')
	classes = os.listdir(tdir)
	nClasses = len(classes)
	# Initialize confusion matrix
	confusion = zeros((nClasses,nClasses)).astype(uint8)
	nCorrect = 0.
	print "Evaluation started..."
	for i in range(nSamples):
		# Obtain ground truth index
		ground_truth = testLabels[i]
		# Get the LBP histogram
		hist = testHists[i]
		# Obtain the indices of NNs
		_, indices = NN.kneighbors(hist.reshape(1,-1))
		indices = squeeze(indices)
		preds = zeros(len(classes))
		if weightedNN:
			# Weighted voting
			weights = zeros(n)
			for j in range(n): 
				weights[j] += 1. / linalg.norm(hist - trainHists[ indices[j], :])
			for j in range(n): 
				preds[ trainLabels[ indices[j] ] ] += weights[j]
		else:
			# Majority voting
			for j in indices: 
				preds[ trainLabels[j] ] += 1
		# print preds
		pred = argmax(preds)
		confusion[ground_truth, pred] += 1
		if pred == ground_truth: nCorrect += 1.
		print "Ground Truth:", classes[ground_truth], "Classification:", classes[pred]
	print "Overall accuracy...", nCorrect / nSamples
	print "Confusion matrix..."
	print confusion

def main():	
	# R = 1
	# P = 8
	n = 5
	# train(R, P)
	# test(R, P)
	evaluate(n, weightedNN=True)

if __name__ == '__main__':
	main()