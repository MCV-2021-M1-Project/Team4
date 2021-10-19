import cv2
import numpy as np

image = cv2.imread('/home/david/Desktop/M1/data/qst2_w1/00000.jpg')
mask = cv2.imread('/home/david/Desktop/M1/Team4/week2/masks/00000.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = mask.astype(np.uint8)

image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([image_color], [0,1,2], mask, [16,16,8], [0,256,0,256,0,256])
hist = cv2.normalize(hist, hist)
print(hist.shape)
