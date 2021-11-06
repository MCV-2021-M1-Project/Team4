import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
import textdistance as td

from text_box import bounding_box
from evaluation import bbox_iou, mapk2paintings
from noise_detection_and_removal import remove_noise
from background_substraction import substractBackground
from read_text import read_text

# ----- WHEN THE BBDD IMAGE AND THE QUERY IMAGE ARE THE SAME -----
n_query_images = 30
with open("../../data/qsd1_w4/text_boxes.pkl", 'rb') as f:
        boxes = pickle.load(f)

iou = np.array([])
text_distance = np.array([])
for i in tqdm(range(n_query_images)): # range(n_query_images)
    img_file = '../../data/qsd1_w4' + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
    img_txt = '../../data/qsd1_w4' + '/00' + ('00' if i < 10 else '0') + str(i) + '.txt'
    
    with open(img_txt,"r") as f:
        data = f.readlines()
        
        text = []
        for d in data:
            text.append(ast.literal_eval(d)[0])
    
    img = cv2.imread(img_file)

    img = remove_noise(img)

    # Obtain the backgorund masks of the image. masks is a list of masks. If there is only a painting in the image
    # the length will be 1, 2 paintings 2, and 3 paintings 3.
    masks = substractBackground(img)

    image_distances = []
    image_bboxes = []
    print(img_file)
    for idx, mask in enumerate(masks):
        # Compute the text box
        """ plt.subplot(231)
        plt.imshow(img) """
        bbox = bounding_box(img, mask=mask)
        """ bbox_mask = np.zeros((img.shape[0], img.shape[1]))
        bbox_mask[mask == 1] = 1
        
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        bbox_mask[top:bottom, left:right] = img_gray[top:bottom, left:right]/255.
        plt.subplot(236)
        plt.imshow(bbox_mask)
        plt.show() """
        
        iou = np.append(iou,bbox_iou(bbox, boxes[i][idx]))
        
        extractedText = read_text(img, bbox)
        print('mask: ' + str(idx))
        print(text[idx])
        print(extractedText)
        
        text_distance = np.append(text_distance,td.levenshtein.distance(extractedText,text[idx]))
        
        
print(np.mean(iou))
print(np.mean(text_distance))