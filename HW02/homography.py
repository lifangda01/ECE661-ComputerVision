#!/usr/bin/python
import numpy as np
import cv2

RESIZE_RATIO = 0.5
BACKGROUND_COLOR = (0, 0, 0)
WORLD_HC = [(0,0), (0,2560), (1536,0), (1536,2560)]
IMAGE_HC_A = [(428,2110), (541,3306), (1496,2150), (1359,3309)]
IMAGE_HC_B = [(796,1558), (771,2983), (1602,1620), (1523,2988)]
IMAGE_HC_C = [(560,1002), (412,2422), (1413,1024), (1482,2404)]
IMAGE_HC_D = [(472,2637), (348,4526), (1468,2646), (1644,4426)]
IMAGE_HC_E = [(492,2846), (300,4408), (1464,2842), (1756,4308)]
IMAGE_HC_F = [(0,0), (0,2591), (1943,0), (1943,2591)]

def get_transformation_matrix(world_hc, image_hc):
	'''
		Given corresponding HC points in world view and image view,
		find the homography matrix between two coordinate systems.
		If sucessful, return a 3x3 numpy matrix
		@world_hc: list of HC point tuples
		@image_hc: list of HC point tuples
		@return: 3x3 numpy matrix H
	'''
	# We need at least for point pairs
	if len(world_hc) != 4 or len(image_hc) != 4:
		print "ERROR: not enough HC point pairs!"
		return None
	# Ah = c --> h = inv(A)c
	A = np.matrix( np.zeros((8,8)) )
	c = np.matrix( np.zeros((8,1)) )
	for i in range( len(world_hc) ):
		(x_w, y_w) = world_hc[i]
		(x_ip, y_ip) = image_hc[i]
		A[2*i] = [x_w, y_w, 1, 0, 0, 0, -x_w*x_ip, -y_w*x_ip]
		A[2*i+1] = [0, 0, 0, x_w, y_w, 1, -x_w*y_ip, -y_w*y_ip]
		c[2*i] = x_ip
		c[2*i+1] = y_ip
	try:
		h = A.I * c
	except np.linalg.LinAlgError:
		print "ERROR: A is singular!"
		return None
	H = np.concatenate( (h,[[1]]), 0)
	return H.reshape(3,3)

def project_world_into_image(world_img, image_img, H):
	'''
		Project world_img into image_img, given their homography H.
		@world_img: np.ndarray of the image to be projected
		@image_img: np.ndarray of the image to be projected on
		@H: the transformation matrix from world plane to image plane
		@return: np.ndarrary of the image with projection
	'''
	proj_img = image_img.copy()
	for r in range(image_img.shape[0]):
		for c in range(image_img.shape[1]):
			p_i = np.array([[r,c,1]])
			(r_w, c_w, z_w) = H.I * p_i.T
			r_w = r_w / z_w
			c_w = c_w / z_w
			if 0 <= r_w < world_img.shape[0]-1 and 0 <= c_w < world_img.shape[1]-1:
				proj_img[r][c] = get_pixel_by_nearest_neighbor(world_img, r_w, c_w)
		print 'row', r
	return proj_img

def transform_image_into_image(image_img_1, image_img_2, H):
	'''
		Transform image_img_1 into the view of image_img_2, given their homography H.
		@image_img_1: np.ndarray of the image to be tranformed
		@image_img_2: np.ndarray of the image with desired view
		@H: the transformation matrix
		@return: np.ndarrary of the transformed image		
	'''
	trans_img = np.ndarray(image_img_2.shape)
	for r in range(trans_img.shape[0]):
		for c in range(trans_img.shape[1]):
			p_t = np.array([[r,c,1]])
			(r_1, c_1, z_1) = H.I * p_t.T
			r_1 = r_1 / z_1
			c_1 = c_1 / z_1
			if 0 <= r_1 < image_img_1.shape[0]-1 and 0 <= c_1 < image_img_1.shape[1]-1:
				trans_img[r][c] = get_pixel_by_nearest_neighbor(image_img_1, r_1, c_1)
			else:
				trans_img[r][c] = BACKGROUND_COLOR
		print 'row', r
	return trans_img		

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

def resize_image_by_ratio(image, ratio):
	'''
		Resize an image by a given ratio, used for faster debug.
	'''
	return cv2.resize(image, (int(image.shape[1]*ratio),int(image.shape[0]*ratio)))

def task1():
	'''
		Transform a front view image onto another image.
	'''
	world_img = cv2.imread('images/Seinfeld.jpg')
	image_img_a = cv2.imread('images/1.jpg')
	image_img_b = cv2.imread('images/2.jpg')
	image_img_c = cv2.imread('images/3.jpg')

	world_img = resize_image_by_ratio(world_img, RESIZE_RATIO)
	image_img_a = resize_image_by_ratio(image_img_a, RESIZE_RATIO)
	image_img_b = resize_image_by_ratio(image_img_b, RESIZE_RATIO)
	image_img_c = resize_image_by_ratio(image_img_c, RESIZE_RATIO)

	world_hc = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in WORLD_HC]
	image_hc_a = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_A]
	image_hc_b = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_B]
	image_hc_c = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_C]

	H_wa = get_transformation_matrix(world_hc, image_hc_a)
	H_wb = get_transformation_matrix(world_hc, image_hc_b)
	H_wc = get_transformation_matrix(world_hc, image_hc_c)

	proj_img = project_world_into_image(world_img, image_img_a, H_wa)
	cv2.imwrite('images/task1_a.jpg', proj_img)
	proj_img = project_world_into_image(world_img, image_img_b, H_wb)
	cv2.imwrite('images/task1_b.jpg', proj_img)
	proj_img = project_world_into_image(world_img, image_img_c, H_wc)
	cv2.imwrite('images/task1_c.jpg', proj_img)

def task2():
	'''
		Cascaded transformations.
	'''
	image_img_a = cv2.imread('images/1.jpg')
	image_img_b = cv2.imread('images/2.jpg')
	image_img_c = cv2.imread('images/3.jpg')

	image_img_a = resize_image_by_ratio(image_img_a, RESIZE_RATIO)
	image_img_b = resize_image_by_ratio(image_img_b, RESIZE_RATIO)
	image_img_c = resize_image_by_ratio(image_img_c, RESIZE_RATIO)

	image_hc_a = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_A]
	image_hc_b = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_B]
	image_hc_c = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_C]

	H_ab = get_transformation_matrix(image_hc_a, image_hc_b)
	H_bc = get_transformation_matrix(image_hc_b, image_hc_c)
	H_ac = H_ab * H_bc

	trans_img = transform_image_into_image(image_img_a, image_img_c, H_ac)
	cv2.imwrite('images/task2.jpg', trans_img)	

def task3():
	'''
		Repeat task 1 and 2 with my own images.
	'''
	image_img_d = cv2.imread('images/4.jpg')
	image_img_e = cv2.imread('images/5.jpg')
	image_img_f = cv2.imread('images/fangda.jpg')

	image_img_d = resize_image_by_ratio(image_img_d, RESIZE_RATIO)
	image_img_e = resize_image_by_ratio(image_img_e, RESIZE_RATIO)
	image_img_f = resize_image_by_ratio(image_img_f, RESIZE_RATIO)

	image_hc_d = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_D]
	image_hc_e = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_E]
	image_hc_f = [( int(x*RESIZE_RATIO), int(y*RESIZE_RATIO) ) for (x,y) in IMAGE_HC_F]

	# Repeat task 1
	# H_fd = get_transformation_matrix(image_hc_f, image_hc_d)
	# H_fe = get_transformation_matrix(image_hc_f, image_hc_e)

	# proj_img = project_world_into_image(image_img_f, image_img_d, H_fd)
	# cv2.imwrite('images/task3_a.jpg', proj_img)
	# proj_img = project_world_into_image(image_img_f, image_img_e, H_fe)
	# cv2.imwrite('images/task3_b.jpg', proj_img)

	# Repeat task 2
	H_de = get_transformation_matrix(image_hc_d, image_hc_e)
	H_ef = get_transformation_matrix(image_hc_e, image_hc_f)
	H_df = H_de * H_ef	

	trans_img = transform_image_into_image(image_img_d, image_img_f, H_df)
	cv2.imwrite('images/task3_c.jpg', trans_img)	

def main():
	# task1()
	# task2()
	task3()

if __name__ == '__main__':
	main()