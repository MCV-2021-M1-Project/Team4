import cv2
import numpy as np
import matplotlib.pyplot as plt
from background_substraction import substractBackground
from text_box import bounding_box
from colorDescriptors import colorDescriptors

img = cv2.imread('/home/david/Desktop/M1/data/qsd2_w2/00000.jpg')

masks = substractBackground(img)

text_boxes = []
text_masks = []

for m in masks:
    bbox = bounding_box(img, m)
    text_boxes.append(bbox)
    empty = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    empty[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    text_masks.append(empty)

joined_masks = []
for m, t in zip(masks, text_masks):
    empty = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    empty[m==1] = 1
    empty[t==1] = 0
    joined_masks.append(empty)

histograms = []
for j in joined_masks:
    histograms = colorDescriptors(img, block=2, mask=j)
