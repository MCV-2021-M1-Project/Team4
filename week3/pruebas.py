import numpy as np

import cv2
from text_box import bounding_box
import matplotlib.pyplot as plt

file = '00015'
dataset = 'qsd2_w2'
img = cv2.imread('../../data/' + dataset + '/' + file + '.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mask = cv2.imread('../../data/' + dataset + '/' + file + '.png')[:,:,0]/255
""" mask[:,int(mask.shape[1]/2):] = 0 """
""" mask[int(mask.shape[0]/2):,:] = 0 """

text_box = bounding_box(img,mask)

