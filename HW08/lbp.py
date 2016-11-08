#!/usr/bin/python
from pylab import *
import cv2
import BitVector

def get_lbp_hist(image, R, P):
	'''
		Find the normalized LBP histogram of an image.
		@image: np.ndarray of gray scale input image
		@R: int of number of LBP sampling circle radius
		@P: int of number of samples to take on each circle
		@return: 1-D array of histogram of the image
	'''
	h, w = image.shape[0], image.shape[1]
	H = zeros(P+2)
	# Iterate through the image
	for r in range(R, h-R):
		for c in range(R, w-R):
			# Obtain the pixel values on the circle
			sample_coords = get_sample_coords((r,c), R, P)
			samples = [image[coord] for coord in sample_coords]
			# samples = [bilinear_interpolate(image, coord[0], coord[1]) for coord in sample_coords]
			samples = array(samples)
			# Obtain the pattern
			pattern = zeros(P)
			pattern[ samples >= image[(r,c)] ] = 1
			# Obtain the encoding and add to histogram
			H[ encode(pattern) ] += 1
			temp.append(encode(pattern))
	return H / float(sum(H))

def bilinear_interpolate(image, y, x):
	'''
		Bilinear interpolation.
	'''
	fx, fy = int(floor(x)), int(floor(y))
	cx, cy = int(ceil(x)), int(ceil(y))
	dx, dy = x-fx, y-fy
	a1 = array([1-dy, dy])
	a2 = array([ [image[fy,fx], image[fy,cx]],
				 [image[cy,fx], image[cy,cx]] ])
	a3 = array([1-dx, dx]) 
	return dot(dot(a1,a2), a3.transpose())

def get_sample_coords(curr, R, P):
	'''
		Given the current pixel coordinate, find the coordinates of samples on the circle.
		@curr: tuple of (r,c)
		@R: int of number of LBP sampling circle radius
		@P: int of number of samples to take on each circle
		@return: list of (r,c) coordinate tuples
	'''
	coords = []
	dTheta = 2*pi / P
	theta = 0.
	for i in range(P):
		dX = np.cos(theta) * R
		dY = np.sin(theta) * R
		new_coord = ( int(curr[0]+0.5+dY), int(curr[1]+0.5+dX) )
		# new_coord = ( curr[0]+0.5+dY, curr[1]+0.5+dX )
		coords.append(new_coord)
		theta += dTheta
	return coords

def encode(pattern):
	'''
		Given pattern, find the integer LBP encoding (0 to P+1).
	'''
	pattern = list(pattern)
	P = len(pattern)
	bv = BitVector.BitVector( bitlist = pattern )
	ints = [int(bv << 1) for _ in range(P)] 
	minbv = BitVector.BitVector( intVal = min(ints), size = P )
	bvruns = minbv.runs()
	if len(bvruns) == 1:
		# Single run of all 0s
		if bvruns[0][0] == 1:
			return 0
		# Single run of all 1s
		else:
			return P
	elif len(bvruns) == 2:
		# 0s followed by 1s
		return len(bvruns[1])
	else:
		# Mixed runs of both 0s and 1s
		return P+1

def main():
	# test = np.array([[5, 4, 2, 4, 2, 2, 4, 0],
	# 		[4, 2, 1, 2, 1, 0, 0, 2],
	# 		[2, 4, 4, 0, 4, 0, 2, 4],
	# 		[4, 1, 5, 0, 4, 0, 5, 5],
	# 		[0, 4, 4, 5, 0, 0, 3, 2],
	# 		[2, 0, 4, 3, 0, 3, 1, 2],
	# 		[5, 1, 0, 0, 5, 4, 2, 3],
	# 		[1, 0, 0, 4, 5, 5, 0, 1]])
	fpath = './imagesDatabaseHW8/training/tree/01.jpg'
	image = imread(fpath)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	print get_lbp_hist(image, 1, 8)

if __name__ == '__main__':
	main()

