import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2

from tqdm import tqdm
from utils import find_mask, bounding_box, evaluation

numImages = 30
TH_V = 75
TH_S = 30

with open('../../data/qsd1_w2/text_boxes.pkl', 'rb') as f:
    data = pickle.load(f)

evaluations = []
for j in tqdm(range(numImages)):
    img_file = '../../data/qsd1_w2/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
    img = cv2.imread(img_file)
    
    """ plt.imshow(img)
    plt.show() """
    
    # RGB to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Value channel
    v = hsv_img[:, :, 2]
    s = hsv_img[:, :, 1]
    
    
    abs_v = np.absolute(v-np.amax(v)/2)
    kernel = np.ones((3,3),np.uint8)
    v_tophat = cv2.morphologyEx(abs_v, cv2.MORPH_BLACKHAT, kernel)
    v_tophat = v_tophat/np.max(v_tophat)
    mask = np.zeros((img.shape[0], img.shape[1]))
    
    tophat_hist = cv2.calcHist([v_tophat], [0], None, [2], [0, 1])
    plt.plot(tophat_hist)
    plt.show()
    mask[v_tophat > 0.4] = 1
    
    kernel = np.ones((3,1),np.uint8)
    mas_closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((2,25),np.uint8)
    mask_open = cv2.morphologyEx(mas_closed, cv2.MORPH_CLOSE, kernel)
    
    plt.subplot(221)
    plt.imshow(v, cmap='gray')
    plt.subplot(222)
    plt.imshow(abs_v, cmap='gray')
    plt.subplot(223)
    plt.imshow(v_tophat, cmap='gray')
    plt.subplot(224)
    plt.imshow(mask_open, cmap='gray')
    plt.show()
    
    """ mask = np.zeros((img.shape[0], img.shape[1]))
    mask[((s< 40) & (v < 60)) | ((s < 22) & (v > 200)) ] = 1
    
    
    kernel = np.ones((21,5),np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_closed = np.array(mask_closed, dtype=np.uint8)
    
    num_comp, components = cv2.connectedComponents(mask_closed)
    bincount = np.bincount(components.flatten())
    bincount_nonzero = np.delete(bincount,0)
    sorted_bins = np.argsort(bincount_nonzero)[::-1]
    
    for cc in range(num_comp - 1):
        max = sorted_bins[cc]
        mask_max = np.zeros((img.shape[0], img.shape[1]))
        mask_max[components == max + 1] = 1
        
        #Width and height of mask
        width = np.count_nonzero(mask_max,axis=0)
        width = np.delete(width,width == 0)
        
        height = np.count_nonzero(mask_max,axis=1)
        height = np.delete(height,height == 0)
        
        if width.shape[0]/height.shape[0] > 1.5 and width.shape[0] > img.shape[1]/3:
            break
        
    boundingbox = bounding_box(mask_max,img)
    
    

    
    # Evaluations
    text_boxes = data[j][0]
    ground_truth = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([text_boxes[0],text_boxes[1],text_boxes[2],text_boxes[3]]), color=1)
    # Evaluation
    ev = evaluation(boundingbox,ground_truth)
    evaluations.append(ev)
    print('image: ' + str(j),ev) """
    
    """ plt.subplot(221)
    plt.imshow(mask,cmap='gray')
    plt.subplot(222)
    plt.imshow(mask_closed, cmap='gray')
    plt.subplot(223)
    plt.imshow(ground_truth, cmap='gray')
    plt.subplot(224)
    plt.imshow(boundingbox,cmap='gray')
    plt.show() """


""" evaluation_mean = np.sum(evaluations, axis=0) / numImages
print()
print("TEXT BOX SUBSTRACTION METHOD:")
print("Precision: {0:.4f}".format(evaluation_mean[0]))
print("Recall: {0:.4f}".format(evaluation_mean[1]))
print("F1-measure: {0:.4f}".format(evaluation_mean[2])) """