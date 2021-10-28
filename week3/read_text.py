import numpy as np
import pytesseract
import cv2
import matplotlib.pyplot as plt
import textdistance as td
import ast

from text_box import bounding_box


def read_text(img,bbox):
    [left, top, right, bottom] = bbox
    
    text_img = img[top:bottom,left:right]  # Crop image
    extractedText = pytesseract.image_to_string(text_img)
    
    # If we don't find text in the image, we try with a bigger bbox
    if extractedText[:-1] == '':
        top = top - 5 if top-5 > 0 else 0
        bottom = bottom + 5 if bottom + 5 < img.shape[0] else 0
        text_img = img[top:bottom,left:right]
        extractedText = pytesseract.image_to_string(text_img)
        
    # If we don't find text in the image, we try with a smaller bbox
    if extractedText[:-1] == '':
        if bottom - top > 20:
            top = top + 10
            bottom = bottom - 10
            text_img = img[top:bottom,left:right]
            extractedText = pytesseract.image_to_string(text_img)

    # Delete extrange chars that never appear in the pictures. Ex.: \n, \r, \x0C, =, %, @, &.
    extractedText = extractedText.replace("\n","").replace("\r","").replace("\x0C","").replace("=","").replace("%","").replace("@","").replace("&","")
    
    # Delete  white spaces in the extrem
    while len(extractedText) and extractedText[0] == ' ':
        extractedText = extractedText[1:]
    
    while len(extractedText) and extractedText[-1] == ' ':
        extractedText = extractedText[:-1]
        
    return extractedText