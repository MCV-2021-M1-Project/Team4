import numpy as np
import cv2
import matplotlib.pyplot as plt
import textdistance as td
import ast

from text_box import bounding_box
from read_text import read_text


distances = np.array([])
distances_letter = np.array([])
for j in range(30):
    file = ('0000' if j < 10 else '000') + str(j)
    print(file)
    dataset = 'qsd2_w2'
    img = cv2.imread('../../data/' + dataset + '/' + file + '.jpg')
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread('../../data/' + dataset + '/' + file + '.png')[:,:,0]/255
    """ mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1])) """
    
    num, mask_components = cv2.connectedComponents(mask.astype(np.uint8))
    
    
    texts = [] 
    with open('../../data/' + dataset + '/' + file + '.txt','r') as f:
        texts = f.readlines()
        """ print(list(text[0][:-1]))
        texts.append(text) """
        
        """ for t in text:
            texts.append(ast.literal_eval(t)[1]) """
    
    
    for cc in range(1,num):
        mask = mask_components == cc
        [left, top, right, bottom] = bounding_box(img_rgb,mask)

        extractedText = read_text(img, [left, top, right, bottom]) 
        distance = td.levenshtein.distance(texts[cc-1][:-1],extractedText)
        print(distance)
        distances = np.append(distances,distance)
        distances_letter = np.append(distances_letter,distance/len(texts[cc-1][:-1]))
    
print(np.mean(distances))
print(np.mean(distances_letter))