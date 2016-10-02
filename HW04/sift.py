import cv2

def get_sift_kp_des(image):
	'''
		Extract SIFT key points and descriptors from image.
		@image: np.ndarray of input image, double type, gray scale
		@return: tuple of key points and corresponding descriptors
	'''
	sift = cv2.xfeatures2d.SIFT_create()
	kps, descs = sift.detectAndCompute(image, None)
	print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
	return kps, descs

