import matplotlib.pyplot as plt
import numpy as np
import cv2

from numpy.core.fromnumeric import mean


from mapk import mapk
from utils import euclidean_distance, l1_distance, chi2_distance, histogram_intersection, hellinger_kernel, borders


img_file = '../../data/qsd2_w1/00001.jpg'
img = cv2.imread(img_file)

transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
channel1 = transformed_img[:,:,1]
channel2 = transformed_img[:,:,2]

mask = np.zeros((channel1.shape[0],channel1.shape[1]))
mask[(channel1[:,:] > 105) | (channel2[:,:] < 40)] = 1

plt.imshow(mask,cmap='gray')
plt.show()

upper_border = np.argmax(mask,axis=0)
min,argmin,max,argmax = borders(upper_border)

pointUL = [argmin,min]
pointUR = [argmax,max]

bottom_border = np.argmax(np.flip(mask),axis=0)
min,argmin,max,argmax = borders(bottom_border)

pointBL = [mask.shape[1] - argmax,mask.shape[0] - max]
pointBR = [mask.shape[1] - argmin,mask.shape[0] - min]

## Get the mask
""" mask = cv2.fillConvexPoly(np.zeros((channel.shape[0],channel.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1) """

## Vis
mask2 = np.zeros((channel1.shape[0],channel1.shape[1]))
points = np.array([pointUL,pointUR,pointBR,pointBL])
mask2 = cv2.line(img,pointUL,pointUR, color=255,thickness =5)
mask2 = cv2.line(mask2,pointUR,pointBR, color=255,thickness =5)
mask2 = cv2.line(mask2,pointBR,pointBL, color=255,thickness =5)
mask2 = cv2.line(mask2,pointBL,pointUL, color=255,thickness =5)
plt.imshow(mask2,cmap='gray')
plt.show()
