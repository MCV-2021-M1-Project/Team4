import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


from utils import inliers
from evaluation import evaluation

def bounds(u):    
    i = inliers(u)

    edges = np.argwhere(i != -1) # Just inliers indexes
    
    left_i = edges.min()
    left_j = u[left_i]
    
    right_i = edges.max()
    right_j = u[right_i]
    
    coordinates = [left_j,left_i,right_j,right_i]
    
    return coordinates

def first_nonzero(arr, axis, invalid_val=-1):
    first_n0 = np.where(arr.any(axis=axis),arr.argmax(axis=axis),invalid_val)
    
    
    if axis==0:
        a = arr[first_n0,np.arange(arr.shape[1])]
        first_n0[a==0] = -1
        
    elif axis == 1:
        a = arr[np.arange(arr.shape[0]),first_n0]
        first_n0[a==0] = -1
    
    return first_n0

def last_nonzero(arr, axis, invalid_val=-1):
    flipped_first_nonzero = first_nonzero(np.flip(arr), axis, invalid_val)
    last_n0 = np.flip(flipped_first_nonzero)
    last_n0[last_n0 != -1] = arr.shape[axis] - last_n0[last_n0 != -1]
    
    return last_n0


#Number of images in the query set
t = 30

#Saturation and Value thresholds
th_s = 114 
th_v = 63


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
    thresholded[(s > th_s) | (v < th_v)] = 1
    
    """ plt.imshow(thresholded, cmap='gray')
    plt.show() """
        

    # Find Upper and Bottom borders
    upper_border = first_nonzero(thresholded, axis=0, invalid_val=-1) # Takes the first non-zero element's index for each array's column        
    bottom_border = last_nonzero(thresholded, axis=0, invalid_val=-1)

    # Find picture's edges coordinates
    if (upper_border > -1).any():
        ul_j,ul_i,ur_j,ur_i = bounds(upper_border)
        bl_j,bl_i,br_j,br_i = bounds(bottom_border)

        pointUL = [ul_i,ul_j] # Upper left point
        pointUR = [ur_i,ur_j] # Upper right point
        pointBL = [bl_i,bl_j] # Bottom left point
        pointBR = [br_i,br_j] # Bottom right point

        ## Draw picture's contours
        """ img_contours = cv2.line(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),pointUL,pointUR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5)
        plt.imshow(img_contours)
        plt.show() """

        ## Get the mask
        mask = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1)
        """ plt.imshow(mask, cmap='gray')
        plt.show() """
        
        plt.imsave('../../data/qsd2_masks/00' + ('00' if i < 10 else '0') + str(i) + '.png',mask,cmap="gray")
    
    """ else:
        mask = np.zeros((img.shape[0],img.shape[1]))

    ground_truth_file = '../../data/qst2_w1/00' + ('00' if i < 10 else '0') + str(i) + '.png'
    ground_truth = cv2.imread(ground_truth_file)
    ground_truth[ground_truth == 255] = 1 # Range [0,255] to [0,1]

    #Evaluation
    evaluations.append(evaluation(mask,ground_truth[:,:,0])) """
    
""" evaluation_mean = np.sum(evaluations,axis=0)/t

print("Precision: " +evaluation_mean[0])
print("Recall: " + evaluation_mean[1])
print("F1-measure: " + evaluation_mean[2]) """