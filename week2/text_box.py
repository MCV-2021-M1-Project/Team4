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

    mask[v_tophat > 0.4] = 1
    
    ##Rellenar letras
    """ kernel = np.ones((1,10),np.uint8) """
    kernel = np.ones((2,10),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #Eliminar linea verticales
    kernel = np.ones((1,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    ##Eliminar lineas horizontales
    kernel = np.ones((4,1),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    ##Juntar letras y palabras
    kernel = np.ones((1,29),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    num_comp, components = cv2.connectedComponents(mask.astype(np.uint8))    

    bincount = np.bincount(components.flatten())
    bincount_nonzero = np.delete(bincount,0)
    
    bincount_sorted = np.argsort(bincount_nonzero)[::-1] + 1
    
    max_cc = np.zeros((img.shape[0], img.shape[1]))
    max_cc[components == bincount_sorted[0]] = 1
    
    limits = np.where(np.amax(max_cc,axis=1))
    limit_sup = limits[0][0]
    limit_inf = limits[0][-1]
    
    inter = limit_inf - limit_sup
    
    limit_sup = limit_sup - inter if limit_sup - inter > 0 else 0
    limit_inf = limit_inf + inter if limit_inf + inter < img.shape[0] else img.shape[0]
    
    cropped_img = np.zeros((img.shape[0], img.shape[1]+202))
    cropped_img[limit_sup:limit_inf,116:cropped_img.shape[1]-116] = v_tophat[limit_sup:limit_inf,15:img.shape[1]-15]
    cropped_img = cropped_img/np.amax(cropped_img)
    
    
    mask = np.zeros((cropped_img.shape[0], cropped_img.shape[1]))

    mask[cropped_img > 0.45] = 1
    
    ##Rellenar letras
    kernel = np.ones((5,14),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #Eliminar linea verticales
    kernel = np.ones((1,4),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    ##Eliminar lineas horizontales
    kernel = np.ones((4,1),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    ##Juntar letras y palabras
    kernel = np.ones((1,90),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    ##Eliminar lineas verticales
    kernel = np.ones((1,2),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    num_comp, components = cv2.connectedComponents(mask.astype(np.uint8))    
    
    bincount = np.bincount(components.flatten())
    bincount_nonzero = np.delete(bincount,0)
    
    bincount_sorted = np.argsort(bincount_nonzero)[::-1] + 1
    
    max_cc_2 = np.zeros((mask.shape[0], mask.shape[1]))
    max_cc_2[components == bincount_sorted[0]] = 1
    

    
    limits_vert = np.where(np.amax(max_cc_2[:,101:-101],axis=1))
    limit_sup = limits_vert[0][0]
    limit_inf = limits_vert[0][-1]
    
    limits_hor = np.where(np.amax(max_cc_2[:,101:-101],axis=0))
    limit_left = limits_hor[0][0]
    limit_right = limits_hor[0][-1]
    
    
    inter = int((limit_inf - limit_sup)*0.5)
    
    limit_sup = limit_sup - inter if limit_sup - inter > 0 else 0
    limit_inf = limit_inf + inter if limit_inf + inter < img.shape[0] else img.shape[0]

    
    limit_left = limit_left - inter if limit_left - inter > 0 else 0
    limit_right = limit_right + inter if limit_right + inter < img.shape[1] else img.shape[1]
    
    cropped_img_2 = np.zeros((img.shape[0], img.shape[1]))
    cropped_img_2[limit_sup:limit_inf,limit_left:limit_right] = 1
    
    pointUL = [limit_left,limit_sup]
    pointUR = [limit_right,limit_sup]
    pointBL = [limit_left,limit_inf]
    pointBR = [limit_right,limit_inf]
    

    """ img_contours = cv2.line(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),pointUL,pointUR, color=255,thickness =5)
    img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
    img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
    img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5) """
    
    text_boxes = data[j][0]
    ground_truth = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([text_boxes[0],text_boxes[1],text_boxes[2],text_boxes[3]]), color=1)
    # Evaluation
    ev = evaluation(cropped_img_2,ground_truth)
    evaluations.append(ev)
    

evaluation_mean = np.sum(evaluations, axis=0) / numImages
print()
print("TEXT BOX SUBSTRACTION METHOD:")
print("Precision: {0:.4f}".format(evaluation_mean[0]))
print("Recall: {0:.4f}".format(evaluation_mean[1]))
print("F1-measure: {0:.4f}".format(evaluation_mean[2]))
    
    
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