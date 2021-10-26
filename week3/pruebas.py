import numpy as np

import cv2
from text_box import bounding_box
import matplotlib.pyplot as plt

img = cv2.imread('00000.jpg')
mask = cv2.imread('00000.png')[:,:,0]/255
mask[:,int(mask.shape[1]/2):-1] = 0

""" mask = np.ones((img.shape[0],img.shape[1])) """
text_box = bounding_box(img,mask)

plt.imshow(text_box,cmap='gray')
plt.show()