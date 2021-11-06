import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract


# TODO:
# Text-boxes are always centered in the picture center
# Text-boces are always in the upper part or the bottom part of the pictures
# Text-boxes must remain inside the picture's mask
# Width is always bigger than height
# Width is at least 1/3*(picture's width)
# Height is at most/ 1/2*(picture's height)

# The process of extracting the bounding box is simple, we iterate 3 times over the same computations.
# Step 1. Compute blackhat of the image region and threshold it
# Step 2. Apply some morphological filter
# Step 3. Find the largest connected component
# Step 4. Crop the original image expanding connected components coordinates.
# Iterate again

def bounding_box(img, mask=None):
    ### PREPROCESSING ###

    height = img.shape[0]
    width = img.shape[1]

    if mask is None:
        mask = np.ones((height, width))

    # Bounding boxes are centered in the picture, so left and right borders have no text.
    # So we compute an auxiliar mask where the right and left borders of the mask are filled with zeros.

    mask_components = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8))  # We take picture's masks' right and left coordinates
    if mask_components[0] == 1:
        mask_left = mask_components[2][0, cv2.CC_STAT_LEFT]
        mask_right = mask_components[2][0, cv2.CC_STAT_LEFT] + mask_components[2][0, cv2.CC_STAT_WIDTH]
        mask_width = mask_components[2][0, cv2.CC_STAT_WIDTH]
        mask_top = mask_components[2][0, cv2.CC_STAT_TOP]
        mask_bottom = mask_components[2][0, cv2.CC_STAT_TOP] + mask_components[2][0, cv2.CC_STAT_HEIGHT]
        mask_height = mask_components[2][0, cv2.CC_STAT_HEIGHT]
    else:
        mask_left = mask_components[2][1, cv2.CC_STAT_LEFT]
        mask_right = mask_components[2][1, cv2.CC_STAT_LEFT] + mask_components[2][1, cv2.CC_STAT_WIDTH]
        mask_width = mask_components[2][1, cv2.CC_STAT_WIDTH]
        mask_top = mask_components[2][1, cv2.CC_STAT_TOP]
        mask_bottom = mask_components[2][1, cv2.CC_STAT_TOP] + mask_components[2][1, cv2.CC_STAT_HEIGHT]
        mask_height = mask_components[2][1, cv2.CC_STAT_HEIGHT]

    mask_aux = mask  # Auxiliar mask
    mask_aux[:, mask_left:mask_left + int(mask_width * 0.12)] = 0
    mask_aux[:, mask_right - int(mask_width * 0.12):mask_right] = 0

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR to HSV
    value = hsv_img[:, :, 2]  # Value channel
    abs_v = np.absolute(value - np.amax(value) / 2)  # Convert blacks to white and maintain whites

    ### END PREPROCESSING ###

    ### 1. ITERATION ###
    ## STEP 1. Blackhat and thresholding
    blackhat = cv2.morphologyEx(abs_v, cv2.MORPH_BLACKHAT, np.ones((3, 3)))
    blackhat = mask_aux * blackhat  # We take just the blackhat values correponding to the auxiliar mask
    blackhat = blackhat / np.max(blackhat)  # Normalize
    expanded_mask = np.zeros(
        (height + 60, width + 22))  # We create an expanded mask to avoid problems in edges with morphological filters
    expanded_mask[30:-30, 11:-11] = blackhat > 0.35  # Thresholding

    ## STEP 2. Morphological filters
    expanded_mask = expanded_mask.astype(np.uint8)
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, np.ones((2, 10)))  # Fill letter
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, np.ones(
        (3, 4)))  # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, np.ones((6, 33)))  # Join letters
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, np.ones((4, 2)))  # Delete remaining vertical lines
    text_mask = expanded_mask[30:-30, 11:-11]  # We reduce the previously expanded mask
    
    """ plt.subplot(232)
    plt.imshow(text_mask) """

    ## STEP 3. Find the biggest connected component's coordinates
    bbox = biggest_component(text_mask)
    if bbox == []:
        bbox = [mask_left, mask_top, mask_right, mask_bottom]
    [left, top, right, bottom] = bbox

    ## STEP 4. Crop the original image --> As detected components represent text and not
    # bounding box, we have to expand top and bottom coordinates to detect the bounding box
    inter = bottom - top
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < height - 10 else height - 10

    blackhat_box = np.zeros((height,
                             width + 200))  # We make the width bigger to avoid problems with morphological filters arriving to edges
    blackhat_box[top:bottom, 130: -130] = blackhat[top:bottom,
                                          30:-30]  # Take original blackhat image's values in that zone. We do not take values from the edges of the image, as textbox i in the center
    blackhat_box = blackhat_box / np.amax(blackhat_box)  # Normalize

    ### 2. ITERATION ###
    ## STEP 1. Blackhat and thresholding
    expanded_mask = np.zeros(
        (height, width + 200))  # We create an expanded mask to avoid problems in edges with morphological filters
    expanded_mask = blackhat_box > 0.3

    ## STEP 2. Morphological filters
    expanded_mask = expanded_mask.astype(np.uint8)
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, np.ones((5, 14)))  # Fill letters
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, np.ones(
        (4, 4)))  # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, np.ones((1, 91)))  # Join letters
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, np.ones((1, 2)))  # Delete remaining vertical lines
    text_mask = expanded_mask[:, 100:-100]  # Reduce the previously expanded mask
    
    """ plt.subplot(233)
    plt.imshow(text_mask) """

    ## STEP 3. Find the biggest connected component's coordinates
    bbox = biggest_component(text_mask)

    if bbox == []:
        bbox = [mask_left, mask_top, mask_right, mask_bottom]
    [left, top, right, bottom] = bbox

    ## STEP 4. Crop the original image --> Expand coordinates and take original image's values in that zone
    inter = int((bottom - top) * 0.5)
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < height - 10 else height - 10
    left = left - inter if left - inter > 30 else 30
    right = right + inter if right + inter < width - 30 else width - 30

    # Some corrections: Text bounding boxes are always centered in the picture.
    # So, we are going to center it
    central_line = mask[top + int((bottom - top) / 2),
                   :]  # We take the row of the mask that lies in the middle of the bounding box
    central_left = np.argmax(central_line)  # We take mask's left column in that row
    central_right = width - np.argmax(central_line[::-1])  # We take mask's right column in that row

    distance_to_text_left = left - central_left  # Left distance from bounding box to mask
    distance_to_text_right = central_right - right  # Right distance from bounding box to mask
    mean_distance = int((
                                    distance_to_text_left + distance_to_text_right) / 2)  # Mean left and right distance from bounding box to the mask

    right = central_right - mean_distance  # Correction in right border
    left = central_left + mean_distance  # Correction in left border

    expanded_mask = np.zeros((height, width + 200))
    expanded_mask[top:bottom, left + 100:right + 100] = blackhat[top:bottom, left:right]  # Cropping

    ### 3. ITERATION ###
    ## STEP 1. Blackhat and thresholding
    expanded_mask = expanded_mask / np.amax(expanded_mask)
    expanded_mask = expanded_mask > 0.3

    ## STEP 2. Morphological filters
    expanded_mask = expanded_mask.astype(np.uint8)
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, np.ones((7, 10)))
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, np.ones((3, 4)))
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, np.ones((3, 87)))
    text_mask = expanded_mask[:, 100:-100]
    
    """ plt.subplot(234)
    plt.imshow(text_mask) """

    ## STEP 3. Find the biggest connected components coordinates
    bbox = biggest_component(text_mask)

    if bbox == []:
        bbox = [0, 0, img.shape[0], img.shape[1]]
    [left, top, right, bottom] = bbox

    ## STEP 4. Crop the original image --> Applying some corrections
    central_line = mask[top + int((bottom - top) / 2), :]
    central_left = np.argmax(central_line)
    central_right = width - np.argmax(central_line[::-1])

    distance_to_text_left = left - central_left
    distance_to_text_right = central_right - right
    if distance_to_text_left > distance_to_text_right:
        left = central_left + distance_to_text_right
    else:
        right = central_right - distance_to_text_left

    height_text = bottom - top
    correction_h = int(height_text * 0.20)
    correction_w = int(height_text * 0.23)
    top = (top - correction_h) if (top - correction_h) > 0 else 0
    bottom = (bottom + correction_h) if (bottom + correction_h) < height else height
    left = left - correction_w if left - correction_w > 0 else 0
    right = right + correction_w if right + correction_w < width else width

    coordinates = [left, top, right, bottom]  # Bounding box's coordinates
    return coordinates


# -- CONNECTED COMPONENTS --

def biggest_component(mask):
    height = mask.shape[0]
    width = mask.shape[1]

    components = cv2.connectedComponentsWithStats(mask)  # Find the connected components

    if components[0] > 1:  # If we have detected at least 1 connected component
        """ print(components[0]) """
        areas = components[2][:, cv2.CC_STAT_WIDTH]  # Take the width of each connected component
        areas = areas[1:]
        index = np.argmax(areas) + 1  # Take the widest connected component

        width_cc = components[2][index, cv2.CC_STAT_WIDTH]  # Components width
        height_cc = components[2][index, cv2.CC_STAT_HEIGHT]  # Components height
        top_cc = components[2][index, cv2.CC_STAT_TOP]  # Components top coordinate

        aspect_ratio = width_cc / height_cc
        not_middle = not (top_cc < int(height / 2) and top_cc + height_cc > int(
            height / 2))  # The component must be in the upper or bottom region of the mask, not in both

        if aspect_ratio > 1.3 and not_middle:
            # Component's coordinates
            left = components[2][index, cv2.CC_STAT_LEFT]  # Left coordinate
            top = components[2][index, cv2.CC_STAT_TOP]  # Top coordinate
            right = left + components[2][index, cv2.CC_STAT_WIDTH]  # Right coordinate
            bottom = top + components[2][index, cv2.CC_STAT_HEIGHT]  # Bottom coordinate
            return left, top, right, bottom
        else:
            mask[components[1] == index] = 0
            return biggest_component(
                mask)  # If conditions are not satisfied, let's try with the next biggest connected component

    else:  # If we don't detect any connected component, the component will be the hole image
        """ left = components[2][0,cv2.CC_STAT_LEFT]
        top = components[2][0,cv2.CC_STAT_TOP]
        right = left + components[2][0,cv2.CC_STAT_WIDTH]
        bottom = top + components[2][0,cv2.CC_STAT_HEIGHT] """
        return []