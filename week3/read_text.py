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
    extractedInformation = pytesseract.image_to_string(text_img)
    
    # If we don't find text in the image, we try with a bigger bbox
    if extractedInformation[:-1] == '':
        top = top - 5 if top-5 > 0 else 0
        bottom = bottom + 5 if bottom + 5 < img.shape[0] else 0
        text_img = img[top:bottom,left:right]
        extractedInformation = pytesseract.image_to_string(text_img)
        
    # If we don't find text in the image, we try with a smaler bbox
    if extractedInformation[:-1] == '':
        if bottom - top > 20:
            top = top + 10
            bottom = bottom - 10
            text_img = img[top:bottom,left:right]
            extractedInformation = pytesseract.image_to_string(text_img)

    # Delete extrange chars that never appear in the pictures. Ex.: \n, \r, \x0C, =, %, @, &.
    extractedInformation = extractedInformation.replace("\n","").replace("\r","").replace("\x0C","").replace("=","").replace("%","").replace("@","").replace("&","")
    
    # Delete  white spaces in the extrem
    while len(extractedInformation) and extractedInformation[0] == ' ':
        extractedInformation = extractedInformation[1:]
    
    while len(extractedInformation) and extractedInformation[-1] == ' ':
        extractedInformation = extractedInformation[:-1]
        
    return extractedInformation