import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

from numpy.core.fromnumeric import mean


from mapk import mapk
from utils import euclidean_distance, l1_distance, chi2_distance, histogram_intersection, hellinger_kernel, borders


img_file = '../../data/qsd2_w1/00030.jpg'
img = cv2.imread(img_file)

transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
channel = transformed_img[:,:,1]
channel2 = transformed_img[:,:,2]
pixel_s = channel[0,0]
interval = 10

mask = np.zeros((channel.shape[0],channel.shape[1]))
mask[(channel[:,:] > 105) | (channel2[:,:] < 40)] = 1
mask[mask == True] = 1
mask[mask == False] = 0

plt.imshow(mask,cmap='gray')
plt.show()

print(mask.shape)

upper_border = np.argmax(mask,axis=0)
min,argmin,max,argmax = borders(upper_border, mask.shape[1])

pointUL = [argmin,min]
pointUR = [argmax,max]

print(pointUL,pointUR)

bottom_border = np.argmax(np.flip(mask),axis=0)
min,argmin,max,argmax = borders(bottom_border, mask.shape[1])

pointBL = [mask.shape[1] - argmax,mask.shape[0] - max]
pointBR = [mask.shape[1] - argmin,mask.shape[0] - min]


""" mask = cv2.fillConvexPoly(np.zeros((channel.shape[0],channel.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1) """
mask2 = np.zeros((channel.shape[0],channel.shape[1]))
points = np.array([pointUL,pointUR,pointBR,pointBL])
mask2 = cv2.line(img,pointUL,pointUR, color=255,thickness =5)
mask2 = cv2.line(mask2,pointUR,pointBR, color=255,thickness =5)
mask2 = cv2.line(mask2,pointBR,pointBL, color=255,thickness =5)
mask2 = cv2.line(mask2,pointBL,pointUL, color=255,thickness =5)
plt.imshow(mask2,cmap='gray')
plt.show() 
""" print(np.argwhere(mask == 1)) """
