#!/usr/bin/env python

##  LBP.py
##  Author:   Avi Kak   (kak@purdue.edu)
##  Date:     NOvember 1, 2016

##  This script was written as a teaching aid for the lecture on "Textures and Color" 
##  as a part of my class on Computer Vision at Purdue.  
##
##  This Python script demonstrates how Local Binary Patterns can be used for 
##  characterizing image textures.

##  For educational purposes, this script generates five different types of textures 
##  -- you make the choice by uncommenting one of the statements in lines (A1) 
##  through (A5).  You can also set the size of the image array and number of gray 
##  levels to use.

##  HOW TO USE THIS SCRIPT:
##
##     1.   Specify the texture type you want by uncommenting one of the lines (A1) 
##          through (A5)
##
##     2.   Set the image size in line (A6)
##
##     3.   Set the number of gray levels in line (A7)
##
##     4.   Choose a value for the circle radius R in line (A8)
#3
##     5.   Choose a value for the number of sampling points on the circle in line (A9).

##  Calling syntax:    LBP.py

import random
import math
import BitVector

##  UNCOMMENT THE TEXTURE TYPE YOU WANT:
texture_type = 'random'                                                           #(A1)
#texture_type = 'vertical'                                                        #(A2)
#texture_type = 'horizontal'                                                      #(A3)
#texture_type = 'checkerboard'                                                    #(A4)
#texture_type = None                                                              #(A5)

IMAGE_SIZE = 8                                                                    #(A6)
#IMAGE_SIZE = 4                                                                   #(A6)
# GRAY_LEVELS = 6                                                                   #(A7)
R = 1                  # the parameter R is radius of the circular pattern        #(A8)
P = 8                  # the number of points to sample on the circle             #(A9)

image = [[5, 4, 2, 4, 2, 2, 4, 0],
		[4, 2, 1, 2, 1, 0, 0, 2],
		[2, 4, 4, 0, 4, 0, 2, 4],
		[4, 1, 5, 0, 4, 0, 5, 5],
		[0, 4, 4, 5, 0, 0, 3, 2],
		[2, 0, 4, 3, 0, 3, 1, 2],
		[5, 1, 0, 0, 5, 4, 2, 3],
		[1, 0, 0, 4, 5, 5, 0, 1]]

# image = [[0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]               #(B1)

# if texture_type == 'random':                                                      #(B2)
#     image = [[random.randint(0,GRAY_LEVELS-1) 
#                     for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]       #(B3)
# elif texture_type == 'diagonal':                                                  #(B4)
#     image = [[GRAY_LEVELS - 1 if (i+j)%2 == 0 else 0 
#                     for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)]       #(B5)
# elif texture_type == 'vertical':                                                  #(B6)
#     image = [[GRAY_LEVELS - 1 if i%2 == 0 else 0                                 
#                     for i in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]       #(B7)
# elif texture_type == 'horizontal':                                                #(B8)
#     image = [[GRAY_LEVELS - 1 if j%2 == 0 else 0                                  
#                     for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)]       #(B9)
# elif texture_type == 'checkerboard':                                              #(B10)
#     image = [[GRAY_LEVELS - 1 if (i+j+1)%2 == 0 else 0 
#                     for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)]       #(B11)
# else:                                                                             #(B12)
#     image = [[1, 5, 3, 1],[5, 3, 1, 4],[4, 0, 0, 0],[2, 3, 4, 5]]                 #(B13)
#     IMAGE_SIZE = 4                                                                #(B14)
#     GRAY_LEVELS = 3                                                               #(B15)

print "Texture type chosen: ", texture_type                                       #(C1)
print "The image: "                                                               #(C2)
for row in range(IMAGE_SIZE): print image[row]                                    #(C3)

lbp = [[0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]                 #(C4)
rowmax,colmax = IMAGE_SIZE-R,IMAGE_SIZE-R                                         #(C5)
lbp_hist = {t:0 for t in range(P+2)}                                              #(C6)

for i in range(R,rowmax):                                                         #(C7)
    for j in range(R,colmax):                                                     #(C8)
        print "\npixel at (%d,%d):" % (i,j), '=', image[i][j]                                       #(C9)                 
        pattern = []                                                              #(C10)
        for p in range(P):                                                        #(C11)
            #  We use the index k to point straight down and l to point to the 
            #  right in a circular neighborhood around the point (i,j). And we 
            #  use (del_k, del_l) as the offset from (i,j) to the point on the 
            #  R-radius circle as p varies.
            del_k,del_l = R*math.cos(2*math.pi*p/P), R*math.sin(2*math.pi*p/P)    #(C12)
            print del_k,del_l
            if abs(del_k) < 0.001: del_k = 0.0                                    #(C13)
            if abs(del_l) < 0.001: del_l = 0.0                                    #(C14)
            k, l =  i + del_k, j + del_l                                          #(C15)
            k_base,l_base = int(k),int(l)                                         #(C16)
            delta_k,delta_l = k-k_base,l-l_base                                   #(C17)
            if (delta_k < 0.001) and (delta_l < 0.001):                           #(C18)
                image_val_at_p = float(image[k_base][l_base])                     #(C19)
            elif (delta_l < 0.001):                                               #(C20)
                image_val_at_p = (1 - delta_k) * image[k_base][l_base] +  \
                                              delta_k * image[k_base+1][l_base]   #(C21)
            elif (delta_k < 0.001):                                               #(C22)
                image_val_at_p = (1 - delta_l) * image[k_base][l_base] +  \
                                              delta_l * image[k_base][l_base+1]   #(C23)
            else:                                                                 #(C24)
                image_val_at_p = (1-delta_k)*(1-delta_l)*image[k_base][l_base] + \
                                 (1-delta_k)*delta_l*image[k_base][l_base+1]  + \
                                 delta_k*delta_l*image[k_base+1][l_base+1]  + \
                                 delta_k*(1-delta_l)*image[k_base+1][l_base]      #(C25)
            if image_val_at_p >= image[i][j]:                                     #(C26)
                pattern.append(1)                                                 #(C27)
            else:                                                                 #(C28)
                pattern.append(0)                                                 #(C29)
        print "pattern: ", pattern                                                #(C30)
        bv =  BitVector.BitVector( bitlist = pattern )                            #(C31)
        intvals_for_circular_shifts  =  [int(bv << 1) for _ in range(P)]          #(C32)
        minbv = BitVector.BitVector( intVal = \
                                  min(intvals_for_circular_shifts), size = P )    #(C33)
        print "minbv: ", minbv                                                    #(C34)
        bvruns = minbv.runs()                                                     #(C35)
        encoding = None
        if len(bvruns) > 2:                                                       #(C36)
            lbp_hist[P+1] += 1                                                    #(C37)
            encoding = P+1                                                        #(C38)
        elif len(bvruns) == 1 and bvruns[0][0] == '1':                            #(C39)
            lbp_hist[P] += 1                                                      #(C40)
            encoding = P                                                          #(C41)
        elif len(bvruns) == 1 and bvruns[0][0] == '0':                            #(C42)
            lbp_hist[0] += 1                                                      #(C43)
            encoding = 0                                                          #(C44)
        else:                                                                     #(C45)
            lbp_hist[len(bvruns[1])] += 1                                         #(C46)
            encoding = len(bvruns[1])                                             #(C47)
        print "encoding: ", encoding                                              #(C48)
print "\nLBP Histogram: ", lbp_hist                                               #(C49)
