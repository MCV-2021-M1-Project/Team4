import numpy as np
import cv2
import matplotlib.pyplot as plt

# TODO: 
# Text-boxes are always centered in the picture center
# Text-boces are always in the upper part or the bottom part of the pictures
# Text-boxes must remain inside the picture's mask
# Width is always bigger than height
# Width is at least 1/3*(picture's width)
# Height is at most/ 1/2*(picture's height)

def bounding_box(img,mask = None):
    height = img.shape[0]
    width = img.shape[1]
    
    if mask is None:
        mask = np.ones((height, width))
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR to HSV
    value = hsv_img[:,:,2]                          # Value channel 
    abs_v = np.absolute(value - np.amax(value) / 2) # Convert blacks to white and maintain whites

    # Blackhat and thresholding
    blackhat = cv2.morphologyEx(abs_v, cv2.MORPH_BLACKHAT, np.ones((3,3)))
    blackhat = mask * blackhat
    blackhat = blackhat / np.max(blackhat)
    text_mask = blackhat > 0.4

    # Morphological filters
    text_mask = text_mask.astype(np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((2,10))) # Fill letter
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((4,4)))   # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((1,29))) # Join letters

    ## Find biggest connected component
    cc = biggest_component(text_mask)

    # Find component's top and bottom coordinates
    top = np.argmax(np.amax(cc, axis=1))
    bottom = height - np.argmax(np.amax(cc, axis=1)[::-1])

    # Expand coordinates
    inter = bottom - top
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < height else height - 10

    #Take original image's values ni that zone
    blackhat_box = np.zeros((height, width + 200)) # We make the width bigger to avoid problems with morphological filters arriving to edges
    blackhat_box[top:bottom, 115: -115] = blackhat[top:bottom, 15:width - 15] # We do not take values from the edges of the image, as textbox i in the center
    blackhat_box = blackhat_box / np.amax(blackhat_box)
    text_mask = blackhat_box > 0.46

    # Morphological filters
    text_mask = text_mask.astype(np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((5,14))) # Fill letters
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((4,4)))   # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((1,91))) # Join letters
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((1,2)))   # Delete remaining vertical lines

    ##Find biggest connected component
    cc = biggest_component(text_mask[:,100:-100])

    # Find component's rectangle's i coordinates
    top = np.argmax(np.amax(cc, axis=1))
    bottom = height - np.argmax(np.amax(cc, axis=1)[::-1])
    left = np.argmax(np.amax(cc, axis=0))
    right = width - np.argmax(np.amax(cc, axis=0)[::-1])

    # Expand coordinates and take original image's values in that zone
    inter = int((bottom - top) * 0.5)
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < height else height - 10
    left = left - inter if left - inter > 0 else 30
    right = right + inter if right + inter < width else width -30
    coordinates = [left, top, right, bottom]

    return coordinates

# -- CONNECTED COMPONENTS --

def biggest_component(mask):
    components = cv2.connectedComponents(mask)[1]               #Find the connected components
    bincount = np.bincount(components[components!=0].flatten()) #Compute the size of each connected component
    index = np.argmax(bincount)                                 #Take the biggest connected component
    
    h = mask.shape[0]
    w = mask.shape[1]
    try:
        new_mask = np.zeros((h, w))
        new_mask[components == index] = 1
    except:
        new_mask = np.ones((h, w))
    return new_mask