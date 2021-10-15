import matplotlib.pyplot as plt
import numpy as np
import cv2

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


# Read image file
t = 30
th_s = 110
th_v = 40
th_s = 114
th_v = 63


values_p = np.array([])
values_r = np.array([])
values_f1 = np.array([])
for th_cb in range(0,1):
    saturations_p = np.array([])
    saturations_r = np.array([])
    saturations_f1 = np.array([])
    for th_s in range(0,260,10):
        evaluations = []
        for i in range(t):
            img_file = '../../data/qsd2_w1/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
            img = cv2.imread(img_file)

            # RGB to HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s = hsv_img[:,:,2]
            """ cb = hsv_img[:,:,2] """
            
            ## Saturation and Value thresholding
            thresholded = np.zeros((img.shape[0],img.shape[1]))
            thresholded[s < th_s] = 1
                

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
            
            else:
                mask = np.zeros((img.shape[0],img.shape[1]))

            ground_truth_file = '../../data/qsd2_w1/00' + ('00' if i < 10 else '0') + str(i) + '.png'
            ground_truth = cv2.imread(ground_truth_file)
            ground_truth[ground_truth == 255] = 1 # Range [0,255] to [0,1]

            #Evaluation
            evaluations.append(evaluation(mask,ground_truth[:,:,0]))
            
        evaluation_mean = np.sum(evaluations,axis=0)/t
        
        saturations_p = np.append(saturations_p,evaluation_mean[0])
        saturations_r = np.append(saturations_r,evaluation_mean[1])
        saturations_f1 = np.append(saturations_f1,evaluation_mean[2])

        print("-------Threshold Value: " + str(th_s) + ", ss: " + str(th_cb) + '-------')
        print("Precision: " + str(evaluation_mean[0]))
        print("Recall: " + str(evaluation_mean[1]))
        print("F1-measure: " + str(evaluation_mean[2]))
        print("\n\n")
        
    
    if values_p.shape[0] == 0:
        values_p = np.array([saturations_p])
        values_r = np.array([saturations_r])
        values_f1 = np.array([saturations_f1])
    else:
        values_p = np.append(values_p,[saturations_p],axis=0)
        values_r = np.append(values_r,[saturations_r],axis=0)
        values_f1 = np.append(values_f1,[saturations_f1],axis=0)
    
    print(values_p)
    
np.save("precision_V.npy",values_p)
np.save("recall_V.npy",values_r)
np.save("f1_measures_V.npy",values_f1)
    
plt.subplot(131)
plt.imshow(values_p)
plt.subplot(132)
plt.imshow(values_r)
plt.subplot(133)
plt.imshow(values_f1)
plt.show()

print('Max. Precision in (S,V):' + str(np.argmax(values_p)))
print('Max. Recall in (S,V):' + str(np.argmax(values_r)))
print('Max. F1-measure in (S,V):' + str(np.argmax(values_f1)))


print(values_p.shape)
for v in range(25):
    plt.plot(np.arange(0,250),values_p[v,:], label = 'Cr: ' + str(v*10))
    
plt.title("Precision")
plt.legend()
plt.show()

for v in range(25):
    plt.plot(np.arange(0,250),values_r[v,:], label = 'Cr: ' + str(v*10))
    
plt.title("Recall")
plt.legend()
plt.show()

for v in range(25):
    plt.plot(np.arange(0,250),values_f1[v,:], label = 'Cr: ' + str(v*10))
    
plt.title("F1-measure")
plt.legend()
plt.show()