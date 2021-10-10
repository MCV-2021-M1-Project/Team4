import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import inliers
from evaluation import evaluation

def bounds(u):    
    i = inliers(u)

    edges = np.argwhere(i != 0) # Just inliers indexes
    
    left_i = edges.min()
    left_j = u[edges.min()]
    
    right_i = edges.max()
    right_j = u[edges.max()]
    
    coordinates = [left_i,left_j,right_i,right_j]
    
    return coordinates

# Read image file
t = 30
th_s = 110
th_v = 40

for th_s in range(100,160):
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
        thresholded[(s[:,:] > th_s) | (v[:,:] < th_v)] = 1

        # Find Upper and Bottom borders
        upper_border = np.argmax(thresholded,axis=0) # Takes the first non-zero element's index for each array's column
        bottom_border = np.argmax(np.flip(thresholded),axis=0)

        # Find picture's edges coordinates
        ul_row,ul_col,ur_row,ur_col = bounds(upper_border)
        br_row_flip,br_col_flip,bl_row_flip,bl_col_flip  = bounds(bottom_border) # flipped coordinates
        
        br_row = img.shape[0] - br_row_flip
        br_col = img.shape[1] - br_col_flip
        bl_row = img.shape[0] - bl_row_flip
        bl_col = img.shape[1] - bl_col_flip

        pointUL = [ul_col,ul_row] # Upper left point
        pointUR = [ur_col,ur_row] # Upper right point
        pointBL = [bl_col,bl_row] # Bottom left point
        pointBR = [br_col,br_row] # Bottom right point

        ## Draw picture's contours
        """ img_contours = cv2.line(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),pointUL,pointUR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5)
        plt.imshow(img_contours)
        plt.show() """

        ## Get the mask
        mask = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1)

        ground_truth_file = '../../data/qsd2_w1/00' + ('00' if i < 10 else '0') + str(i) + '.png'
        ground_truth = cv2.imread(ground_truth_file)
        ground_truth[ground_truth == 255] = 1 # Range [0,255] to [0,1]

        #Evaluation
        evaluations.append(evaluation(mask,ground_truth[:,:,0]))
        
    evaluation_mean = np.sum(evaluations,axis=0)/t

    print("-------Threshold Saturation: " + str(th_s) + ", Value: " + str(th_v) + '-------')
    print("Precision: " + str(evaluation_mean[0]))
    print("Recall: " + str(evaluation_mean[1]))
    print("F1-measure: " + str(evaluation_mean[2]))
    print("\n\n")