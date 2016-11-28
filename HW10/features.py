import cv2

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

def get_sift_matchings(kp1, des1, kp2, des2, rowDiff=10, maxDist=300):
	'''
		Get matchings from SIFT features with constrains and return list of matched points.
	'''
	bf = cv2.BFMatcher()
	matches = bf.match(des1, des2)
	pts1, pts2 = [], []
	good = []
	for m in matches:
		pt1 = kp1[m.queryIdx].pt
		pt2 = kp2[m.trainIdx].pt
		# Reinforce constraint from rectification
		# Matches have to reside in approximately the same row
		if abs(pt1[1] - pt2[1]) > rowDiff or m.distance > maxDist: continue
		pts1.append(pt1)
		pts2.append(pt2)
		good.append([m])
	return pts1, pts2, good	