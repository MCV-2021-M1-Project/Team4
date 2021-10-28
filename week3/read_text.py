import numpy as np
import pytesseract
import cv2
import matplotlib.pyplot as plt
import textdistance as td
import ast

from text_box import bounding_box


def read_text(img,bbox):
    
    text_img = preprocessing(img,bbox)

    extractedText = pytesseract.image_to_string(img)
    
    # If we don't find text in the bbox, we try with a bigger bbox
    
    [left, top, right, bottom] = bbox
    if extractedText[:-1] == '':
        top = top - 5 if top-5 > 0 else 0
        bottom = bottom + 5 if bottom + 5 < img.shape[0] else img.shape[0]
        new_bbox = [left, top, right, bottom]
    
        text_img = preprocessing(text_img, new_bbox)
        
        extractedText = pytesseract.image_to_string(text_img)
        
    # If we don't find text in the bbox, we try with a smaller bbox
    if extractedText[:-1] == '':
        if bottom - top > 20:
            top = top + 10
            bottom = bottom - 10
            new_bbox = [left, top, right, bottom]

            text_img = preprocessing(text_img,new_bbox)
            
            extractedText = pytesseract.image_to_string(text_img)

    # Delete extrange chars that never appear in the pictures. Ex.: \n, \r, \x0C, =, %, @, &, #, «, ¢, “
    extractedText = extractedText.replace("\n","").replace("\r","").replace("\x0C","").replace("=","").replace("%","").replace("@","").replace("&","").replace("#","").replace("«","").replace("¢","").replace('“','')
    
    # Delete white spaces in the extrem
    while len(extractedText) and extractedText[0] == ' ':
        extractedText = extractedText[1:]
    
    while len(extractedText) and extractedText[-1] == ' ':
        extractedText = extractedText[:-1]
    
    print(extractedText)
    return extractedText

def preprocessing(img,bbox):
    [left, top, right, bottom] = bbox
    
    cropped_img = img[top:bottom,left:right]  # Crop image
    cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
    
    correction = 30
    hist = cv2.calcHist(cropped_img,[0],None,[256],[0,256])
    max = np.argmax(hist)
    
    if max > 127:
        binarized_img = cropped_img > (max - correction)
    else:
        binarized_img = cropped_img < (max + correction)
        
    return binarized_img