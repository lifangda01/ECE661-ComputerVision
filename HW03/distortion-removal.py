#!/usr/bin/python
import numpy as np
import cv2

RESIZE_RATIO = 1.0
BACKGROUND_COLOR = (0, 0, 0)
# line(1,2) is parallel to line(3,4)
IMAGE_HC_A = [(190,120), (69,540), (236,108), (120,546)]

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
		@return: (num_row, num_col, off_row, off_col)
	'''
	(h, w, c) = image.shape
	corners_1 = [(0,0), (0,w), (h,0), (h,w)]
	corners_2_row = []
	corners_2_col = []
	for corner in corners_1:
		(r,c) = corner
		p_1 = np.array([[r,c,1]])
		(r_2, c_2, z_2) = H * p_1.T
		corners_2_row.append( int(r_2 / z_2) )
		corners_2_col.append( int(c_2 / z_2) )
	return (max(corners_2_row)-min(corners_2_row)+1, max(corners_2_col)-min(corners_2_col)+1,
			min(corners_2_row), min(corners_2_col))

def get_line(p1, p2):
	p1 = np.array([p1[0], p1[1], 1.])
	p2 = np.array([p2[0], p2[1], 1.])
	return np.cross(p1, p2)

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

def get_projective_transformation_matrix(line1, line2, line3, line4):
	'''
		Given two pairs of lines that are supposed to be parallel in world plane,
		find the homography matrix.
		If successful, return a 3x3 numpy matrix that can be used to remove projective distortion.
		@line1..4: line 1,2 and line 3,4 should be parallel in world plane,
				   each line is a tuple of its HC representation
		@return: 3x3 numpy matrix H
	'''
	# Calculate the two vanishing points first
	vp1 = np.cross(line1, line2)
	vp2 = np.cross(line3, line4)
	# Calculate the vanishing line
	vl = np.cross(vp1, vp2)
	vl = vl / vl[2]
	H = np.matrix(np.identity(3))
	H[2] = vl
	return H

def get_affine_transformation_matrix(line1, line2, line3, line4):
	'''
		Given two pairs of lines that are supposed to be orthogonal in world plane,
		find the homography matrix.
		If successful, return a 3x3 numpy matrix that can be used to remove affine distortion.
		@line1..4: line 1,2 and line 3,4 should be orthogonal in world plane,
				   each line is a tuple of its HC representation
		@return: 3x3 numpy matrix H
	'''	
	# Get S first
	# As = c --> s = inv(A)c
	M = np.matrix( np.zeros((2,2)) )
	c = np.matrix( np.zeros((2,1)) )
	M[0] = [line1[0]*line2[0], line1[0]*line2[0] + line1[1]*line2[0]]
	M[1] = [line3[0]*line4[0], line3[0]*line4[0] + line3[1]*line4[0]]
	c[0] = -line1[1]*line2[1]
	c[1] = -line3[1]*line4[1]
	try:
		s = M.I * c
	except np.linalg.LinAlgError:
		print "ERROR: M is singular!"
		return None	
	s = np.asarray(s).reshape(-1)
	S = np.array([[s[0], s[1]],
				   [s[1], 1.  ]])
	# Perform eigen value decomposition
	U, s, V = np.linalg.svd(S)
	D = np.sqrt(np.diag(s))
	H = np.matrix( np.zeros((3,3)) )
	H[:2, :2] = (U * D * V)
	H[:2, :2] = H[:2, :2] / H[1,1]
	H[2,2] = 1.
	print H
	return H

def get_affine_transformation_matrix_from_points(points):
	return get_affine_transformation_matrix(
			get_line(points[0], points[1]), get_line(points[0], points[2]),
			get_line(points[3], points[1]), get_line(points[3], points[2]) )

def get_projective_transformation_matrix_from_points(points):
	return get_projective_transformation_matrix(
			get_line(points[0], points[1]), get_line(points[2], points[3]),
			get_line(points[0], points[2]), get_line(points[1], points[3]) )

def apply_transformation(image, H):
	'''
		Given a transformation matrix, apply it to the input image to obtain a transformed image.
		@image: np.ndarray of input image
		@H: the tranformation matrix to be applied
		@return: np.ndarray of the transformed image
	'''
	# First determine the size of the transformed image 
	(num_row, num_col, off_row, off_col) = get_bounding_box_after_transformation(image, H)
	print (num_row, num_col, off_row, off_col)
	try:
		trans_img = np.ndarray( (num_row, num_col, image.shape[2]) )
	except IndexError:
		trans_img = np.ndarray( (num_row, num_col, 1) )
	H_inv = H.I
	for r in range(trans_img.shape[0]):
		for c in range(trans_img.shape[1]):
			p_2 = np.array([[r+off_row,c+off_col,1]])
			(r_1, c_1, z_1) = H_inv * p_2.T
			r_1 = r_1 / z_1
			c_1 = c_1 / z_1
			if 0 <= r_1 < image.shape[0]-1 and 0 <= c_1 < image.shape[1]-1:
				trans_img[r][c] = get_pixel_by_nearest_neighbor(image, r_1, c_1)
			else:
				trans_img[r][c] = BACKGROUND_COLOR		
	return trans_img	

def task1():
	'''
		Use two-step method to remove affine distortion
	'''
	image_img_a = cv2.imread('images/flatiron.jpg')
	# image_img_b = cv2.imread('images/monalisa.jpg')
	# image_img_c = cv2.imread('images/wideangle.jpg')

	image_img_a = resize_image_by_ratio(image_img_a, RESIZE_RATIO)
	# image_img_b = resize_image_by_ratio(image_img_b, RESIZE_RATIO)
	# image_img_c = resize_image_by_ratio(image_img_c, RESIZE_RATIO)

	image_hc_a = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_A]
	# image_hc_b = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_B]
	# image_hc_c = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_C]

	H_ap = get_projective_transformation_matrix_from_points(image_hc_a)
	H_aa = get_affine_transformation_matrix_from_points(image_hc_a)

	trans_img = apply_transformation(image_img_a, H_aa.I * H_ap)
	cv2.imwrite('images/task1_a.jpg', trans_img)

	# proj_img = project_world_into_image(world_img, image_img_b, H_wb)
	# cv2.imwrite('images/task1_b.jpg', proj_img)
	# proj_img = project_world_into_image(world_img, image_img_c, H_wc)
	# cv2.imwrite('images/task1_c.jpg', proj_img)	

def main():
	# line1 = np.array([-1,5,-20])
	# line2 = np.array([2,-4,38])
	# line3 = np.array([-8,2,30])
	# line4 = np.array([-5,-1,56])
	# print get_transformation_matrix(line1, line2, line3, line4)
	task1()

if __name__ == '__main__':
	main()