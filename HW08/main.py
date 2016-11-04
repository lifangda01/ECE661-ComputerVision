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
			fpath = os.path.join(sdir,s)
			image = imread(fpath)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			hist = get_lbp_hist(image, R, P)
			if hists != None:
				hists = vstack( (hists, hist) )
				labels = append(labels, label)
			else:
				hists = hist
				labels = array([label])
			print "Training finished on...", fpath
	savetxt("hists.out", hists)
	savetxt("labels.out", hists)
	print "Trained histogram and labels saved..."

def test(R, P, k):
	'''
		Evaluate our classifier.
		@R: int of number of LBP sampling circle radius
		@P: int of number of samples to take on each circle
		@k: int of number of nearest neighbors for voting
	'''
	trainHists = loadtxt("hists.out")
	trainLabels = loadtxt("labels.out")
	NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(trainHists)
	tdir = os.path.join(os.getcwd(),'imagesDatabaseHW8','training')
	classes = os.listdir(tdir)
	tdir = os.path.join(os.getcwd(),'imagesDatabaseHW8','testing')
	samples = os.listdir(tdir)
	print "Testing started..."
	for s in samples:
		fpath = os.path.join(tdir,s)
		image = imread(fpath)
		print "Testing...", fpath
		ground_truth = classes.index( s.split("_")[0] )
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		hist = get_lbp_hist(image, R, P)
		_, index = NN.kneighbors(hist.reshape(1,-1))
		index = index[0][0]
		print "Ground Truth:", ground_truth, "Classification:", classes[index]

def main():	
	# train(1, 8)
	test(1, 8, 5)

if __name__ == '__main__':
	main()