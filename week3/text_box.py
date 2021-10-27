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
    text_mask = np.zeros((height + 60, width + 22))
    text_mask[30:-30,11:-11] = blackhat > 0.4

    # Morphological filters
    text_mask = text_mask.astype(np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((2,10))) # Fill letter
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((4,4)))   # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((1,29))) # Join letters
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((4,2)))   # Delete remaining vertical lines
    text_mask = text_mask[30:-30,11:-11]
    
    ## Find biggest connected component's coordinates
    [left, top, right, bottom] = biggest_component(text_mask)

    # Expand coordinates
    inter = bottom - top
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < height else height - 10

    #Take original image's values in that zone
    blackhat_box = np.zeros((height, width + 200)) # We make the width bigger to avoid problems with morphological filters arriving to edges
    blackhat_box[top:bottom, 115: -115] = blackhat[top:bottom, 15:width - 15] # We do not take values from the edges of the image, as textbox i in the center
    blackhat_box = blackhat_box / np.amax(blackhat_box)
    text_mask = blackhat_box > 0.5
    
    """ plt.imshow(text_mask,cmap='gray')
    plt.show() """

    # Morphological filters
    text_mask = text_mask.astype(np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((5,14))) # Fill letters
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((4,4)))   # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((1,91))) # Join letters
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((1,2)))   # Delete remaining vertical lines

    ##Find biggest connected component's coordinates
    [left, top, right, bottom] = biggest_component(text_mask[:,100:-100])

    # Expand coordinates and take original image's values in that zone
    inter = int((bottom - top) * 0.5)
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < height else height - 10
    left = left - inter if left - inter > 0 else 30
    right = right + inter if right + inter < width else width -30
    
    
    # Center bounding_box
    central_line = mask[top + int((bottom - top)/2),:]
    central_left = np.argmax(central_line)
    central_right = width - np.argmax(central_line[::-1])
    
    distance_to_text_left = left - central_left
    distance_to_text_right = central_right - right
    mean_distance = int((distance_to_text_left + distance_to_text_right)/2)
    
    right = central_right - mean_distance
    left = central_left + mean_distance
    
    point1 = [left,top]
    point2 = [right,top]
    point3 = [right,bottom]
    point4 = [left,bottom]
    
    text_mask = cv2.fillConvexPoly(np.zeros((height, width)),np.array([point1,point2,point3,point4]), color=1)    
    
    coordinates = [left, top, right, bottom]

    return coordinates

# -- CONNECTED COMPONENTS --

def biggest_component(mask):
    height = mask.shape[0]
    width = mask.shape[1]
    
    components = cv2.connectedComponentsWithStats(mask) #Find the connected components
    areas = components[2][:,cv2.CC_STAT_WIDTH]           #Take the area of each connected component
    areas = areas[1:]
    index = np.argmax(areas) + 1                        #Take the biggest connected component
    
    width_cc = components[2][index,cv2.CC_STAT_WIDTH]
    height_cc = components[2][index,cv2.CC_STAT_HEIGHT]
    top_cc = components[2][index,cv2.CC_STAT_TOP]
    
    if width_cc/height_cc > 1.3 and not (top_cc < int(height/2) and top_cc + height_cc > int(height/2)):
        # Component's coordinates
        left = components[2][index,cv2.CC_STAT_LEFT]
        top = components[2][index,cv2.CC_STAT_TOP]
        right = left + components[2][index,cv2.CC_STAT_WIDTH]
        bottom = top + components[2][index,cv2.CC_STAT_HEIGHT]
        return left, top, right, bottom
    else:
        mask[components[1] == index] = 0
        return biggest_component(mask)
    