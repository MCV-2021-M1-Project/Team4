import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import borders
from evaluation import evaluation


# Read image file
t = 30

evaluations = []
for i in range(t):
    img_file = '../../data/qsd2_w1/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
    img = cv2.imread(img_file)

    # RGB to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv_img[:,:,1]
    v = hsv_img[:,:,2]

    ## Saturation and Value thresholding
    thresholded = np.zeros((img.shape[0],img.shape[1]))
    thresholded[(s[:,:] > 110) | (v[:,:] < 40)] = 1

    # Find Upper and Bottom borders
    upper_border = np.argmax(thresholded,axis=0)
    min,argmin,max,argmax = borders(upper_border)

    pointUL = [argmin,min]
    pointUR = [argmax,max]

    bottom_border = np.argmax(np.flip(thresholded),axis=0)
    min,argmin,max,argmax = borders(bottom_border)

    pointBL = [img.shape[1] - argmax,img.shape[0] - max]
    pointBR = [img.shape[1] - argmin,img.shape[0] - min]

    ## Draw picture's contours
    img_contours = cv2.line(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),pointUL,pointUR, color=255,thickness =5)
    img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
    img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
    img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5)
    """ plt.imshow(img_contours,cmap='gray')
    plt.show() """

    ## Get the mask
    mask = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1)

    ground_truth_file = '../../data/qsd2_w1/00' + ('00' if i < 10 else '0') + str(i) + '.png'
    ground_truth = cv2.imread(ground_truth_file)
    ground_truth[ground_truth == 255] = 1 # Range [0,255] to [0,1]

    #Evaluation
    evaluations.append(evaluation(mask,ground_truth[:,:,0]))
    
evaluation_mean = np.sum(evaluations,axis=0)/t

print("Precision: " + str(evaluation_mean[0]))
print("Recall: " + str(evaluation_mean[1]))
print("F1-measure: " + str(evaluation_mean[2]))