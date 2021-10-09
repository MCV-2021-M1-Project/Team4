import matplotlib.pyplot as plt
import numpy as np
import cv2

from numpy.core.fromnumeric import mean


from mapk import mapk
from utils import euclidean_distance, l1_distance, chi2_distance, histogram_intersection, hellinger_kernel, borders


# Read image file
img_file = '../../data/qsd2_w1/00001.jpg'
img = cv2.imread(img_file)

# RGB to HSV
transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
channel1 = transformed_img[:,:,1]
channel2 = transformed_img[:,:,2]

## Saturation and Value threshold
thresholded = np.zeros((img.shape[0],img.shape[1]))
thresholded[(channel1[:,:] > 105) | (channel2[:,:] < 40)] = 1

plt.imshow(thresholded,cmap='gray')
plt.show()

# Find Upper and Bottom borders
upper_border = np.argmax(thresholded,axis=0)
min,argmin,max,argmax = borders(upper_border)

pointUL = [argmin,min]
pointUR = [argmax,max]

bottom_border = np.argmax(np.flip(thresholded),axis=0)
min,argmin,max,argmax = borders(bottom_border)

pointBL = [img.shape[1] - argmax,img.shape[0] - max]
pointBR = [img.shape[1] - argmin,img.shape[0] - min]

## Get the mask
""" mask = cv2.fillConvexPoly(np.zeros((channel.shape[0],channel.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1) """

## Draw picture's contours
img_contours = cv2.line(img,pointUL,pointUR, color=255,thickness =5)
img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5)
plt.imshow(img_contours,cmap='gray')
plt.show()
