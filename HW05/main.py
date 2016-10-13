#!/usr/bin/python
import numpy as np
import cv2
from ransac import apply_ransac_on_matchings
from llsm import get_llsm_homograhpy_from_points
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def get_bounding_box_after_transformation(image, H):
	'''
		Given an image and the transformation matrix to be applied, 
		calculate the bounding box of the image after transformation.
		@image: image to transform
		@H: transformation matrix to be applied
		@return: (h, w, oy, ox)
	'''
	h, w = image.shape[0], image.shape[1]
	corners_1 = [(0,0), (w,0), (0,h), (w,h)]
	corners_2_x = []
	corners_2_y = []
	H = np.matrix(H)
	for corner in corners_1:
		(x,y) = corner
		p_1 = np.array([[x,y,1]])
		(x_2, y_2, z_2) = H * p_1.T
		corners_2_x.append( int(x_2 / z_2) )
		corners_2_y.append( int(y_2 / z_2) )
	return max(corners_2_y)-min(corners_2_y)+1, max(corners_2_x)-min(corners_2_x)+1, \
			min(corners_2_y), min(corners_2_x)

def get_pixel_by_nearest_neighbor(image, row_f, col_f):
	'''
		Get the pixel value based on float row and column numbers.
		@image: image to be find pixels in
		@row_f, col_f: float row and column numbers
		@return: pixel value from image
	'''
	row = int(round(row_f))
	col = int(round(col_f))
	return image[row][col]

def apply_transformation_on_image(image, H):
	'''
		Given a transformation matrix, apply it to the input image to obtain a transformed image.
		@image: np.ndarray of input image
		@H: the tranformation matrix to be applied
		@return: np.ndarray of the transformed image
	'''
	# First determine the size of the transformed image 
	h, w, oy, ox = get_bounding_box_after_transformation(image, H)
	print "New image size:", (h, w)
	print "Offsets:", oy, ox
	try:
		trans_img = np.ndarray( (h, w, image.shape[2]) )
	except IndexError:
		trans_img = np.ndarray( (h, w, 1) )
	H = np.matrix(H)
	H_inv = H.I
	for y in range(trans_img.shape[0]):
		for x in range(trans_img.shape[1]):
			p_2 = np.array([[x+ox,y+oy,1]])
			(x_1, y_1, z_1) = H_inv * p_2.T
			x_1 = x_1 / z_1
			y_1 = y_1 / z_1
			if 0 <= y_1 < image.shape[0]-1 and 0 <= x_1 < image.shape[1]-1:
				trans_img[y][x] = get_pixel_by_nearest_neighbor(image, y_1, x_1)
			else:
				trans_img[y][x] = tuple(np.zeros(trans_img.shape[2]))		
	return trans_img

def get_canvas(cen_img, sur_imgs, Hs):
	(h, w, c) = cen_img.shape
	minx, miny, maxx, maxy = 0, 0, w, h
	for i in range(len(sur_imgs)):
		h, w, oy, ox = get_bounding_box_after_transformation(sur_imgs[i], Hs[i])
		minx, miny, maxx, maxy = min(minx, ox), min(miny, oy), \
								max(maxx, ox+w), max(maxy, oy+h)
	return np.zeros( (maxy-miny, maxx-minx, c) ), minx, miny

def get_sift_kp_des(image, nfeatures=0):
	'''
		Extract SIFT key points and descriptors from image.
		@image: np.ndarray of input image, double type, gray scale
		@return: tuple of key points and corresponding descriptors
	'''
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
	kps, descs = sift.detectAndCompute(image, None)
	print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
	return kps, descs

def get_matchings(kp1, des1, kp2, des2):
	'''
		Get matchings using SIFT and return non-openCV points
	'''
	bf = cv2.BFMatcher()
	matches = bf.match(des1, des2)
	pts1, pts2 = [], []
	for m in matches:
		pt1 = kp1[m.queryIdx].pt
		pt2 = kp2[m.trainIdx].pt
		pts1.append(pt1)
		pts2.append(pt2)
	return pts1, pts2	

def test_ransac():
	fpath1 = 'images/6.jpg'
	fpath2 = 'images/7.jpg'
	resize_ratio = 0.5
	nfeatures = 1000
	epsilon = 0.4
	delta = 10
	color1 = cv2.imread(fpath1)
	color1 = resize_image_by_ratio(color1, resize_ratio)
	gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
	color2 = cv2.imread(fpath2)
	color2 = resize_image_by_ratio(color2, resize_ratio)
	gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	kp1, des1 = get_sift_kp_des(gray1, nfeatures=nfeatures)
	kp2, des2 = get_sift_kp_des(gray2, nfeatures=nfeatures)
	pts1, pts2 = get_matchings(kp1, des1, kp2, des2)
	in1, in2 = apply_ransac_on_matchings(pts1, pts2, epsilon, delta)
	H = get_llsm_homograhpy_from_points(in1, in2)
	sur_imgs = [color1]
	Hs = [H]
	canvas, canvas_ox, canvas_oy = get_canvas(color2, sur_imgs, Hs)
	print canvas.shape, canvas_ox, canvas_oy
	warped1 = apply_transformation_on_image(color1, H)

	h,w = color2.shape[0], color2.shape[1]
	canvas[ 0-canvas_oy : 0-canvas_oy+h , 0-canvas_ox : 0-canvas_ox+w ] = color2
	h,w,oy,ox = get_bounding_box_after_transformation(color1, H)
	canvas[ oy-canvas_oy : oy-canvas_oy+h , ox-canvas_ox : ox-canvas_ox+w ] = warped1

	plt.imshow(cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB), cmap='jet'), plt.show()
	
	# plt.imshow(cv2.cvtColor(warped1.astype(np.uint8), cv2.COLOR_BGR2RGB), cmap='jet'), plt.show()
	# plt.imshow(warped1), plt.show()
	# Plot image and mark the corners
	# fig, axes = plt.subplots(1,2)
	# axes[0].set_aspect('equal')
	# axes[0].imshow(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB), cmap='jet')
	# axes[1].set_aspect('equal')
	# axes[1].imshow(cv2.cvtColor(color2, cv2.COLOR_BGR2RGB), cmap='jet')
	# for i in range(len(pts1)):
	# 	color = np.random.rand(3,1)
	# 	pt1 = pts1[i]
	# 	pt2 = pts2[i]
	# 	axes[0].add_patch( Circle(pt1, 5, fill=False, color=color, clip_on=False) )
	# 	axes[1].add_patch( Circle(pt2, 5, fill=False, color=color, clip_on=False) )
	# 	# Draw lines for matching pairs
	# 	line1 = ConnectionPatch(xyA=pt1, xyB=pt2, coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1], color=color)
	# 	line2 = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA='data', coordsB='data', axesA=axes[1], axesB=axes[0], color=color)
	# 	axes[0].add_patch(line1)
	# 	axes[1].add_patch(line2)
	# plt.show()

def show_sift():
	fpath1 = 'images/1.jpg'
	fpath2 = 'images/2.jpg'
	resize_ratio = 0.5
	color1 = cv2.imread(fpath1)
	color1 = resize_image_by_ratio(color1, resize_ratio)
	gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
	color2 = cv2.imread(fpath2)
	color2 = resize_image_by_ratio(color2, resize_ratio)
	gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	kp1, des1 = get_sift_kp_des(gray1, nfeatures=1000)
	kp2, des2 = get_sift_kp_des(gray2, nfeatures=1000)
	# Find the matchings
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append([m])
	# Weird fix for cv2.drawMatchesKnn error
	img3 = np.zeros((1,1))
	# cv2.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(cv2.cvtColor(color1, cv2.COLOR_BGR2RGB),kp1,
							cv2.cvtColor(color2, cv2.COLOR_BGR2RGB),kp2,
							good,img3,flags=2)
	plt.imshow(img3),plt.show()

def main():
	test_ransac()
	# show_sift()


if __name__ == '__main__':
	main()