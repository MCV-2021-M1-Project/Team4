import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage import feature

from noise_detection_and_removal import remove_noise

TH_S = 114  # Saturation threshold
TH_V = 63   # Value threshold

# FILE WITH ALL THE FUNCTIONS NEEED TO SUBSTRACT THE BACKGROUND.

def substractBackground(image, plot = False):
    """
    :param image: Image in BGR
    :return: List of masks.
            If no painting is detected a list with a np.ones(image.shape) is returned.
            If one painting is detected a list with one mask is retruned.
            If two painitngs are detected a list with two masks is returned.
    """

    # Transform the BGR image into HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels (only s and v are used later on)
    h, s, v = cv2.split(hsv_image)

    # Threshold the image with the TH_S and TH_V values and create an initial mask
    thresholded_hsv = np.zeros((image.shape[0], image.shape[1]))
    thresholded_hsv[(s > TH_S) | (v < TH_V)] = 1

    # Input: mask
    # Output: list with masks of the lists.
    components = connected_components(image, thresholded_hsv, plot = plot)

    image_masks = []
    for cc in components:
        image_masks.append(cc)

    return image_masks


# -- AUXILIARY FUNCTIONS TO SUBTRACT BACKGROUND --

def connected_components(image, mask, plot):
    # THREE TYPES OF MASK ARE USED: CLOSING, GRADIENT AND CANNY
    # THEN, THE UNION OF THE THREE IS DONE

    image_area = mask.shape[0] * mask.shape[1]  # Area of the image in pixels

    if plot:
        f,ax = plt.subplots(2,3)
        ax[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0,0].title.set_text('Original image')

    # ----- 1. CLOSING -----

    # Closing with a very small structuring element to remove random points of the mask
    kernel = np.ones((10, 10), np.uint8)
    closing_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing of 50 x 50 to put the interior of the paintings as mask (value 1)
    kernel = np.ones((50, 50), np.uint8)
    closing_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_CLOSE, kernel)

    # Transform mask to unsigned int8 due to compatibility with open cv functions
    closing_mask = np.array(closing_mask, dtype=np.uint8)

    # Find the connected components of the CLOSING MASK
    # num_comp: number of components detected (CAREFUL!!! THE BACKGROUND COUNTS AS A CONNECTED COMPONENT)
    # components: same mask as the input but with different values for each connected component
    # stats: list of statistics of each components such as:
    #        stats[i][cv2.CC_STAT_TOP]:     top pixel of the component
    #        stats[i][cv2.CC_STAT_LEFT]:    left pixel of the component
    #        stats[i][cv2.CC_STAT_WIDTH]:   width of the component
    #        stats[i][cv2.CC_STAT_HEIGHT]:  height of the component
    #        stats[i][cv2.CC_STAT_AREA]:    area in pixels of the component
    # centroids (_): centroids of the component (NOT USED)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(closing_mask)

    # Obtain the BBox coordinates of each component
    props = regionprops(components)

    # for each component draw the BBox full if two conditions are ensured (value 1)
    # p1: the are of the BBox has to be bigger than the 4% of the image
    # p2: if the BBox is next to a border, the BBox has to be bigger than the half of height
    closing_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for prop in props:
        p1 = ((prop.bbox[3] - prop.bbox[1]) * (prop.bbox[2] - prop.bbox[0]) / image_area) > 0.04
        p2 = prop.bbox[1] == 0 and (prop.bbox[3] - prop.bbox[1]) < (image.shape[0] / 2)
        if p1 and not p2:
            cv2.rectangle(closing_mask, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), 1, -1)

    # Find again the connected components of the CLOSING MASK (after changing each components for its full BBox)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(closing_mask)

    # Remove the component if it has the same width or height as the image
    joined_closing = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx, s in enumerate(stats):
        if idx != 0 and (s[cv2.CC_STAT_WIDTH] < mask.shape[1] and s[cv2.CC_STAT_HEIGHT] < mask.shape[0]):
            joined_closing[components == idx] = 1

    closing_mask = joined_closing

    if plot:
        ax[0,1].imshow(closing_mask)
        ax[0,1].title.set_text('Closing mask')

    # ----- 2. MORPHOLOGIC GRADIENT -----

    # As the image has been denoised, apply a sharpener filter to the mask to enhance and sharpen edges
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    mask = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel)

    # Perform the morphological gradient by doing erosion and dilation separately and then subtract
    # (with small structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    gradient_mask = dilation - erosion

    # Apply closing of 75 x 75 to propagate the big gradient values (sharp edges)
    kernel = np.ones((75, 75), np.uint8)
    gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE, kernel)

    # Threshold the mask and change the dynamic range to [0, 1]
    (T, gradient_mask) = cv2.threshold(gradient_mask, 0.5, 1, cv2.THRESH_BINARY)

    # Transform mask to unsigned int8 due to compatibility with open cv functions
    gradient_mask = np.array(gradient_mask, dtype=np.uint8)

    # Find the connected components of the GRADIENT MASK
    # num_comp: number of components detected (CAREFUL!!! THE BACKGROUND COUNTS AS A CONNECTED COMPONENT)
    # components: same mask as the input but with different values for each connected component
    # stats: list of statistics of each components such as:
    #        stats[i][cv2.CC_STAT_TOP]:     top pixel of the component
    #        stats[i][cv2.CC_STAT_LEFT]:    left pixel of the component
    #        stats[i][cv2.CC_STAT_WIDTH]:   width of the component
    #        stats[i][cv2.CC_STAT_HEIGHT]:  height of the component
    #        stats[i][cv2.CC_STAT_AREA]:    area in pixels of the component
    # centroids (_): centroids of the component (NOT USED)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(gradient_mask)

    props = regionprops(components)

    # for each component draw the BBox full if two conditions are ensured (value 1)
    # p1: the are of the BBox has to be bigger than the 4% of the image
    # p2: if the BBox is next to a border, the BBox has to be bigger than the half of height
    gradient_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for prop in props:
        p1 = ((prop.bbox[3] - prop.bbox[1]) * (prop.bbox[2] - prop.bbox[0]) / image_area) > 0.04
        p2 = prop.bbox[1] == 0 and (prop.bbox[3] - prop.bbox[1]) < (image.shape[0] / 2)
        if p1 and not p2:
            cv2.rectangle(gradient_mask, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), 1, -1)

    # Find again the connected components of the GRADIENT MASK (after changing each components for its full BBox)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(gradient_mask)

    # Remove the component if it has the same width or height as the image
    joined_gradient = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx, s in enumerate(stats):
        if idx!=0 and (s[cv2.CC_STAT_WIDTH] < mask.shape[1] and s[cv2.CC_STAT_HEIGHT] < mask.shape[0]):
            joined_gradient[components == idx] = 1

    gradient_mask = joined_gradient

    if plot:
        ax[0,2].imshow(gradient_mask)
        ax[0,2].title.set_text('gradient mask')

    # ----- 3. CANNY -----

    # Transform the image to gray level and use canny detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_mask = np.uint8(feature.canny(gray, sigma=1))

    # Closing of 30 x 30 to put the interior of the paintings as mask (value 1)
    kernel = np.ones((30, 30), np.uint8)
    canny_mask = cv2.morphologyEx(canny_mask, cv2.MORPH_CLOSE, kernel)
    # Opening to avoid random points
    kernel = np.ones((10, 10), np.uint8)
    canny_mask = cv2.morphologyEx(canny_mask, cv2.MORPH_OPEN, kernel)

    # Find the connected components of the GRADIENT MASK
    # num_comp: number of components detected (CAREFUL!!! THE BACKGROUND COUNTS AS A CONNECTED COMPONENT)
    # components: same mask as the input but with different values for each connected component
    # stats: list of statistics of each components such as:
    #        stats[i][cv2.CC_STAT_TOP]:     top pixel of the component
    #        stats[i][cv2.CC_STAT_LEFT]:    left pixel of the component
    #        stats[i][cv2.CC_STAT_WIDTH]:   width of the component
    #        stats[i][cv2.CC_STAT_HEIGHT]:  height of the component
    #        stats[i][cv2.CC_STAT_AREA]:    area in pixels of the component
    # centroids (_): centroids of the component (NOT USED)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(canny_mask)

    # Obtain the BBox coordinates of each component
    props = regionprops(components)

    # for each component draw the BBox full if two conditions are ensured (value 1)
    # p1: the are of the BBox has to be bigger than the 4% of the image
    # p2: if the BBox is next to a border, the BBox has to be bigger than the half of height
    canny_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for prop in props:
        p1 = ((prop.bbox[3] - prop.bbox[1]) * (prop.bbox[2] - prop.bbox[0]) / image_area) > 0.04
        p2 = prop.bbox[1] == 0 and (prop.bbox[3] - prop.bbox[1]) < (image.shape[0] / 2)
        if p1 and not p2:
            cv2.rectangle(canny_mask, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), 1, -1)

    if plot:
        ax[1,0].imshow(canny_mask)
        ax[1,0].title.set_text('canny mask')

    # ----- 4. UNION OF BOTH MASKS -----

    # Make the union of both masks
    union_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    union_mask[(closing_mask == 1)] = 1
    union_mask[(gradient_mask == 1)] = 1
    union_mask[(canny_mask == 1)] = 1

    # Find again the connected components of the UNION MASK
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(union_mask)

    # Obtain the BBox coordinates of each component
    props = regionprops(components)

    # for each component draw the BBox full if two conditions are ensured (value 1)
    for prop in props:
        cv2.rectangle(components, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), 1, -1)

    union_mask = components.astype(np.uint8)

    if plot:
        ax[1,1].imshow(union_mask)
        ax[1,1].title.set_text('union mask')

    # ----- 5. MASK SEPARATION -----

    # Separate the connected components into different masks
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(union_mask)

    # If the number of the connected components is 1, which means that no painting has been detected. Create a mask of 1
    if num_comp == 1:
        union_mask = np.ones(mask.shape[:2], dtype=np.uint8)

    # If the number of connected components is more than 1 (background and one painting or more), iterate on the
    # connected components. Store the component in a new mask if the component is:
    #   Not the position 0 (as it is the background)
    #   Same width or length as the original image
    #   Area of the component is bigger than a threshold
    else:
        union_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for idx, s in enumerate(stats):
            if idx!=0 and (s[cv2.CC_STAT_WIDTH] < mask.shape[1] or s[cv2.CC_STAT_HEIGHT] < mask.shape[0]) and (s[cv2.CC_STAT_AREA] / image_area) > 0.04:
                union_mask[components == idx] = 1

    # The border of the image is never a painting, so add 0 (For rotation purposes)
    union_mask[0,:] = 0
    union_mask[:,0] = 0
    union_mask[mask.shape[0]-1,:] = 0
    union_mask[:, mask.shape[1]-1] = 0

    # Order the paintings from left to right
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(union_mask)

    lefts = []
    for idx, s in enumerate(stats):
        if idx!= 0:
            lefts.append(s[cv2.CC_STAT_LEFT])

    order = np.argsort(np.array(lefts)).tolist()

    comp = []   # List in which all the masks will be appended
    if num_comp == 1:
        comp.append(np.ones(mask.shape[:2], dtype=np.uint8))

    else:
        for pos in order:
            comp_i = np.zeros((mask.shape[0], mask.shape[1]))
            comp_i[components == pos+1] = 1
            comp.append(np.array(comp_i, dtype=np.uint8))

    if plot:
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        for c in comp:
            empty[c==1] = 1
        ax[1,2].imshow(empty)
        ax[1,2].title.set_text('separated mask')
        plt.show()

    return comp
