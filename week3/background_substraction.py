import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    # TWO TYPES OF MASK: CLOSING AND GRADIENT

    # ----- 1. CLOSING -----

    # Apply closing of 63 x 63 to put the interior of the paintings as mask (value 1)
    kernel = np.ones((50, 50), np.uint8)
    closing_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Transform mask to unsigned int8 to compattibility with open cv functions
    closing_mask = np.array(closing_mask, dtype=np.uint8)

    # Find the number of connected components of the mask (num_comp)
    # The following function returns:
    # - num_comp:   Number of connected components
    # - components: The same mask as the input but each component
    #               has a different value from 0 to num_components ordered from bigger to smaller.
    # - stats: Statistics of each connected component, including BBox and area in pixels
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(closing_mask)

    image_area = mask.shape[0] * mask.shape[1]  # Area of the image in pixels
    maxComp = 0  # Counter of the connected components. When maxComp == 2 -> STOP
    comp = []  # List with the mask of the connected components

    # Search the two biggest components. If some anomalies occur empty the components list to discard
    # this mask
    # The anomalies are:
    # - If the connected components has the same width or height as the original image
    # - If the BBox of the connected components is in one of the corners
    # If there is no anomalies and the percentage of the area of the components is bigger than 0.02
    # append it to the list
    # STOP when there are already 2 components in the list
    for idx, s in enumerate(stats):
        if idx != 0 and (s[cv2.CC_STAT_WIDTH] == mask.shape[1] or s[cv2.CC_STAT_HEIGHT] == mask.shape[0]):
            comp = []
            break

        if idx != 0 and ((s[cv2.CC_STAT_TOP] == 0 and s[cv2.CC_STAT_LEFT] == 0) or ((s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_WIDTH]) ==  mask.shape[1] and (s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_HEIGHT]) == mask.shape[0])):
            comp = []
            break

        if (s[cv2.CC_STAT_WIDTH] * s[cv2.CC_STAT_HEIGHT] / image_area) < 1 and (
                (s[cv2.CC_STAT_AREA] / image_area) > 0.02):
            comp_i = np.zeros((mask.shape[0], mask.shape[1]))
            comp_i[components == idx] = 1
            comp.append(np.array(comp_i, dtype=np.uint8))
            maxComp = maxComp + 1

            if maxComp == 2:
                break

    # Concatenate the 2 biggest components
    if len(comp) == 0:
        closing_mask = np.zeros((mask.shape[0], mask.shape[1]))
    if len(comp) == 1:
        closing_mask = np.zeros((mask.shape[0], mask.shape[1]))
        closing_mask[(comp[0]==1)] = 1
    if len(comp) == 2:
        closing_mask = np.zeros((mask.shape[0], mask.shape[1]))
        closing_mask[(comp[0]==1) | (comp[1]==1)] = 1

    # ----- 2. MORPHOLOGIC GRADIENT + CLOSING -----

    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Perform the morphological gradient aby doing erosion and dilation separately and then subtract
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    gradient_mask = dilation - erosion

    # Apply closing of 63 x 63 to put the interior of the paintings as mask (value 1)
    kernel = np.ones((30, 63), np.uint8)
    gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE, kernel)

    # Threshold the mask and change range to [0-1]
    (T, gradient_mask) = cv2.threshold(gradient_mask, 0.5, 1, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    gradient_mask = cv2.erode(gradient_mask, kernel, iterations=1)
    gradient_mask = cv2.dilate(gradient_mask, kernel, iterations=1)
    gradient_mask = np.array(gradient_mask, dtype=np.uint8)

    # Find the number of connected components of the mask (num_comp)
    # The following function returns:
    # - num_comp:   Number of connected components
    # - components: The same mask as the input but each component
    #               has a different value from 0 to num_components ordered from bigger to smaller.
    # - stats: Statistics of each connected component, including BBox and area in pixels
    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(gradient_mask)
    image_area = mask.shape[0] * mask.shape[1]  # Area of the image in pixels
    maxComp = 0  # Counter of the connected components. When maxComp == 2 -> STOP
    comp = []  # List with the mask of the connected components

    # Search the two biggest components. If some anomalies occur empty the components list to discard
    # this mask
    # The anomalies are:
    # - If the connected components has the same width or height as the original image
    # - If the BBox of the connected components is in one of the corners
    # If there is no anomalies and the percentage of the area of the components is bigger than 0.02
    # append it to the list
    # STOP when there are already 2 components in the list
    for idx, s in enumerate(stats):
        if idx != 0 and (s[cv2.CC_STAT_WIDTH] == mask.shape[1] or s[cv2.CC_STAT_HEIGHT] == mask.shape[0]):
            comp = []
            break

        if idx != 0 and s[cv2.CC_STAT_AREA] / image_area > 0.02 and (s[cv2.CC_STAT_TOP] == 0 or s[cv2.CC_STAT_LEFT] == 0 or (s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_WIDTH]) == mask.shape[1] or (s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_HEIGHT]) == mask.shape[0]):
            comp = []
            break

        if (s[cv2.CC_STAT_WIDTH] * s[cv2.CC_STAT_HEIGHT] / image_area) < 1 and (
                s[cv2.CC_STAT_AREA] / image_area) > 0.02:
            comp_i = np.zeros((mask.shape[0], mask.shape[1]))
            comp_i[components == idx] = 1
            comp.append(np.array(comp_i, dtype=np.uint8))
            maxComp = maxComp + 1

            if maxComp == 2:
                break
    # Concatenate the 2 biggest components
    if len(comp) == 0:
        gradient_mask = np.zeros((mask.shape[0], mask.shape[1]))
    if len(comp) == 1:
        gradient_mask = np.zeros((mask.shape[0], mask.shape[1]))
        gradient_mask[(comp[0] == 1)] = 1
    if len(comp) == 2:
        gradient_mask = np.zeros((mask.shape[0], mask.shape[1]))
        gradient_mask[(comp[0] == 1) | (comp[1] == 1)] = 1

    # Union of the 2 masks
    union_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    union_mask[(closing_mask==1) | (gradient_mask==1)] = 1

    # Separate the masks in a list.
    # If there are no paintings return one empty mask.
    # If there is 1 painting return a list with 1 mask
    # If there is 2 painting return a list with 2 mask

    (num_comp, components, stats, _) = cv2.connectedComponentsWithStats(union_mask)

    separated_masks = []

    if num_comp == 1:
        separated_masks.append(np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8))

    elif num_comp == 2:
        components_i = np.zeros((mask.shape[0], mask.shape[1]))
        components_i[components == 1] = 1
        separated_masks.append(np.array(components_i, dtype=np.uint8))

    else:
        for idx in range(1, 3):
            components_i = np.zeros((mask.shape[0], mask.shape[1]))
            components_i[components == idx] = 1
            separated_masks.append(np.array(components_i, dtype=np.uint8))

    return separated_masks

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






