import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops
from noise_detection_and_removal import remove_noise

TH_S = 114  # Saturation threshold
TH_V = 63   # Value threshold

# FILE WITH ALL THE FUNCTIONS NEEED TO SUBSTRACT THE BACKGROUND.

def substractBackground(image):
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
    thresholded_image = np.zeros((image.shape[0], image.shape[1]))
    thresholded_image[(s > TH_S) | (v < TH_V)] = 1

    # Input: mask
    # Output: list with masks of the lists. If there
    components = connected_components(thresholded_image)

    image_masks = []
    for cc in range(len(components)):
        image_masks.append(find_mask(components[cc]))

    return image_masks


# -- AUXILIARY FUNCTIONS TO SUBTRACT BACKGROUND --

def connected_components(mask):
    # TWO TYPES OF MASK ARE USED: CLOSING AND GRADIENT
    # THEN, THE UNION OF THE TWO IS DONE

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

    # for each component draw the BBox full (value 1)
    for prop in props:
        cv2.rectangle(components, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), 1, -1)

    closing_mask = components.astype(np.uint8)

    # Find again the connected components of the CLOSING MASK (after changing each components for its full BBox)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(closing_mask)

    # Remove the component if it has the same width or height as the image
    joined_gradient = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx, s in enumerate(stats):
        if idx != 0 and (s[cv2.CC_STAT_WIDTH] < mask.shape[1] and s[cv2.CC_STAT_HEIGHT] < mask.shape[0]):
            joined_gradient[components == idx] = 1

    closing_mask = joined_gradient

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

    # Obtain the BBox coordinates of each component
    props = regionprops(components)
    for prop in props:
        cv2.rectangle(components, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), 1, -1)

    gradient_mask = components.astype(np.uint8)

    # Find again the connected components of the GRADIENT MASK (after changing each components for its full BBox)
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(gradient_mask)

    # Remove the component if it has the same width or height as the image
    joined_gradient = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx, s in enumerate(stats):
        if idx!=0 and (s[cv2.CC_STAT_WIDTH] < mask.shape[1] and s[cv2.CC_STAT_HEIGHT] < mask.shape[0]):
            joined_gradient[components == idx] = 1

    gradient_mask = joined_gradient

    # ----- 3. UNION OF BOTH MASKS -----

    # Make the union of both masks
    union_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    union_mask[(closing_mask == 1)] = 1
    union_mask[(gradient_mask == 1)] = 1

    # Find again the connected components of the UNION MASK
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(union_mask)

    # ----- 4. MASK SEPARATION -----
    # Separate the connected components into different masks

    comp = []   # List in which all the masks will be appended

    # If the number of the connected components is 1, which means that no painting has been detected. Create a mask of 1
    if num_comp == 1:
        comp.append(np.ones(mask.shape[:2], dtype=np.uint8))

    # If the number of connected components is more than 2 (background and one painting or more), iterate on the
    # connected components. Store the component in a new mask if the component is:
    #   Not the position 0 (as it is the background)
    #   Same width or length as the original image
    #   Area of the component is bigger than a threshold
    else:
        image_area = mask.shape[0] * mask.shape[1]  # Area of the image in pixels
        for idx, s in enumerate(stats):
            if idx!=0 and (s[cv2.CC_STAT_WIDTH] < mask.shape[1] or s[cv2.CC_STAT_HEIGHT] < mask.shape[0]) and (s[cv2.CC_STAT_AREA] / image_area) > 0.05 :
                comp_i = np.zeros((mask.shape[0], mask.shape[1]))
                comp_i[components == idx] = 1
                comp.append(np.array(comp_i, dtype=np.uint8))

    return comp

def find_mask(connected_component):
    # Find Upper and Bottom borders
    # Takes the first non-zero element's index for each array's column
    upper_border = first_nonzero(connected_component, axis=0)
    bottom_border = last_nonzero(connected_component, axis=0)

    # Find picture's edges coordinates
    if (upper_border > -1).any():
        ul_j, ul_i, ur_j, ur_i = bounds(upper_border)
        bl_j, bl_i, br_j, br_i = bounds(bottom_border)

        pointUL = [ul_i, ul_j]  # Upper left point
        pointUR = [ur_i, ur_j]  # Upper right point
        pointBL = [bl_i, bl_j]  # Bottom left point
        pointBR = [br_i, br_j]  # Bottom right point

        # Get the mask and convert it to unit8 to not have problems in cv2.CalcHist function later
        mask = cv2.fillConvexPoly(np.zeros((connected_component.shape[0], connected_component.shape[1])),
                                  np.array([pointUL, pointUR, pointBR, pointBL]), color=1)

        return mask.astype(np.uint8)

    else:
        mask = np.zeros((connected_component.shape[0], connected_component.shape[1]), dtype="uint8")

        return mask

def first_nonzero(arr, axis):
    first_n0 = np.where(arr.any(axis=axis), arr.argmax(axis=axis), -1)

    if axis == 0:
        a = arr[first_n0, np.arange(arr.shape[1])]

    elif axis == 1:
        a = arr[np.arange(arr.shape[0]), first_n0]

    first_n0[a == 0] = -1

    return first_n0

def last_nonzero(arr, axis):
    flipped_first_nonzero = first_nonzero(np.flip(arr), axis)
    last_n0 = np.flip(flipped_first_nonzero)
    last_n0[last_n0 != -1] = arr.shape[axis] - last_n0[last_n0 != -1]

    return last_n0

def bounds(u):
    i = inliers(u)
    edges = np.argwhere(i != -1)  # Just inliers indexes

    left_i = edges.min()
    left_j = u[left_i]

    right_i = edges.max()
    right_j = u[right_i]

    coordinates = [left_j, left_i, right_j, right_i]

    return coordinates

def inliers(u):
    # Detected border's must be close to each other
    upper_bound, bottom_bound = inliers_bounds(np.extract(u != -1, u))

    # Inliers
    inliers = u
    inliers[u > upper_bound] = -1
    inliers[u < bottom_bound] = -1

    return inliers

def inliers_bounds(u):
    q1 = np.quantile(u, 0.25)  # First quantile
    q3 = np.quantile(u, 0.75)  # Second quantile
    q_inter = q3 - q1  # Interquantile interval

    # Inliers bounds
    upper_bound = q3 + 1.5 * q_inter
    bottom_bound = q1 - 1.5 * q_inter

    return upper_bound, bottom_bound

# -- AUXILIARY FUNCTIONS EVALUATE THE BACKGROUND SUBTRACTION--

def evaluation(predicted, truth):
    tp = np.zeros(predicted.shape)
    fp = np.zeros(predicted.shape)
    fn = np.zeros(predicted.shape)

    tp[(predicted[:, :] == 1) & (truth[:, :] == 1)] = 1
    fp[(predicted[:, :] == 1) & (truth[:, :] == 0)] = 1
    fn[(predicted[:, :] == 0) & (truth[:, :] == 1)] = 1

    p = precision(tp, fp)
    r = recall(tp, fn)
    f1 = f1_measure(p, r)

    return p, r, f1

def precision(tp, fp):
    return np.nan_to_num(np.sum(tp) / (np.sum(tp) + np.sum(fp)))

def recall(tp, fn):
    return np.nan_to_num(np.sum(tp) / (np.sum(tp) + np.sum(fn)))

def f1_measure(p, r):
    return np.nan_to_num(2 * p * r / (p + r))
