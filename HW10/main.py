#!/usr/bin/python
from pylab import *
import cv2
from rectification import *
from features import *
from mpl_toolkits.mplot3d import Axes3D

def pick_points(image):
	'''
		Prompt user to click points on image.
	'''
	points = []
	fig = plt.figure()
	def onclick(event):
		if len(points) == 8:
			return
		x = int(event.xdata)
		y = int(event.ydata)
		points.append((x,y))
		print "Mouse clicked at (x,y)...", (x,y), "; Total number of points clicked...", len(points)
		print points
	imshow(image)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	show()

def main():
	nfeatures = 0
	w,h = 1000, 1000
	image1 = imread('./images/1.jpg')
	image2 = imread('./images/2.jpg')
	# subplot(1,2,1)
	# imshow(image1)
	# subplot(1,2,2)
	# imshow(image2)
	# Uncomment to pick points if necessary
	# pick_points(image1)
	# pick_points(image2)
	pts1_manual = [(92, 112), (290, 142), (131, 263), (245, 296), (151, 326), (151, 365), (101, 493), (277, 428)]
	pts2_manual = [(39, 142), (257, 115), (66, 259), (176, 310), (80, 315), (81, 352), (52, 441), (248, 503)]
	F = get_fundamental_matrix(pts1_manual, pts2_manual)
	print "F = ", F
	e, ep = get_epipoles(F)
	print "e = ", e
	print "ep = ", ep
	P, Pp = get_canonical_projection_matrices(F, ep)
	print "P = ", P
	print "Pp = ", Pp
	print "======== First Nonlinear Optimization ========"
	P_refined, Pp_refined = nonlinear_optimization(pts1_manual, pts2_manual, P, Pp)
	print "P_refined = ", P_refined
	print "Pp_refined = ", Pp_refined
	F_refined = get_fundamental_matrix_from_projection(P_refined, Pp_refined)
	print "F_refined = ", F_refined
	e_refined, ep_refined = get_epipoles(F_refined)
	print "e_refined = ", e_refined
	print "ep_refined = ", ep_refined
	print "======== Rectification ========"
	H, Hp =	get_rectification_homographies(image1, image2, pts1_manual, pts2_manual, e_refined, ep_refined, P_refined, Pp_refined)
	print "H = ", H
	print "Hp = ", Hp
	print "e = ", dot(H, e_refined)
	print "ep = ", dot(Hp, ep_refined)
	rectified1 = cv2.warpPerspective(image1, H, (w,h))
	rectified2 = cv2.warpPerspective(image2, Hp, (w,h))
	print "======== Feature Matching ========"
	kp1, des1 = get_sift_kp_des(rectified1, nfeatures=nfeatures)
	kp2, des2 = get_sift_kp_des(rectified2, nfeatures=nfeatures)
	pts1_ft, pts2_ft, good = get_sift_matchings(kp1, des1, kp2, des2)
	matchings = cv2.drawMatchesKnn(rectified1,kp1,rectified2,kp2,good,array([]),flags=2)
	figure()
	imshow(matchings)
	print "======== Final Nonlinear Optimization ========"
	P_final, Pp_final = nonlinear_optimization(pts1_ft, pts2_ft, P_refined, Pp_refined)
	print "P_final = ", P_final
	print "Pp_final = ", Pp_final
	F_final = get_fundamental_matrix_from_projection(P_final, Pp_final)
	print "F_final = ", F_final
	e_final, ep_final = get_epipoles(F_final)
	print "e_final = ", e_final
	print "ep_final = ", ep_final
	print "======== Final Triangulation ========"
	pts_world = triangulate_points(P_final, Pp_final, pts1_ft, pts2_ft)
	pts_world = array(pts_world)
	fig = figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pts_world[:,0], pts_world[:,1], pts_world[:,2], c='b')
	# Also plot the original manual points
	pts1_manual_rect = apply_transformation_on_points(pts1_manual, H)
	pts2_manual_rect = apply_transformation_on_points(pts2_manual, Hp)
	pts_manual_world = triangulate_points(P_final, Pp_final, pts1_manual_rect, pts2_manual_rect)
	pts_manual_world = array(pts_manual_world)
	ax.scatter(pts_manual_world[:,0], pts_manual_world[:,1], pts_manual_world[:,2], c='r')

	# figure()
	# subplot(1,2,1)
	# imshow(rectified1)
	# subplot(1,2,2)
	# imshow(rectified2)
	show()

if __name__ == '__main__':
	main()