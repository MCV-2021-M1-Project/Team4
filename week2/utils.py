import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt

# -- CHECK ARGUMENTS --

def checkArguments(args):
    if args.m != 'd' and args.m != 't':
        raise TypeError("The modes of the code are: development(d) and test(t)")
    if args.c not in ["GRAY", "RGB", "HSV", "YCrCb", "CIELab"]:
        raise TypeError("Wrong color space")
    if args.d not in ["all", "euclidean", "intersec", "l1", "chi2", "chi2alt2", "hellinger", "chi2alt"]:
        raise TypeError("Wrong distance")
    if args.m == 't' and args.d == 'all':
        raise Exception("The test mode cannot be done with all the distances at the same time given that only one "
                        "distance can be stored in the same file")

# -- PREPROCESSING --

def equalizeImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    eqV = cv2.equalizeHist(v)
    img = cv2.merge((h, s, eqV))
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


# -- SIMILARITY MEASURES --

def euclidean_distance(u,v):
    return np.linalg.norm(u - v)

def l1_distance(u,v):
    return np.linalg.norm((u - v),ord=1)

def chi2_distance(u,v, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(u, v)])

def chi2alternative_distance(u,v):
    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR_ALT)

def histogram_intersection(u,v):
    # return np.sum(np.minimum(u,v))
    return cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)

def hellinger_kernel(u,v):
    """
    # return np.sum(np.sqrt(np.multiply(u,v)))
    n = len(u)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(u[i]) - np.sqrt(v[i])) ** 2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)

# -- IMAGE RETRIEVAL FUNCTIONS --

def computeHistImage(image, color_space, mask=None):
    if color_space == "GRAY":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image_color], [0], mask, [16], [0, 256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "RGB":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Already BGR
        hist = cv2.calcHist([image_color], [0,1,2], mask, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "HSV":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image_color], [0,1,2], mask, [16,16,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "YCrCb":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hist = cv2.calcHist([image_color], [0,1,2], mask, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "CIELab":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        hist = cv2.calcHist([image_color], [0,1,2], mask, [8,16,16], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)

    # plt.plot(image_hist)
    # plt.show()

    return hist.flatten()

def computeSimilarity(hist1, hist2, similarity_measure):
    if similarity_measure == 'euclidean':
        return utils.euclidean_distance(hist1, hist2)
    elif similarity_measure == 'intersec':
        return utils.histogram_intersection(hist1, hist2)
    elif similarity_measure == 'l1':
        return utils.l1_distance(hist1, hist2)
    elif similarity_measure == 'chi2':
        return utils.chi2_distance(hist1, hist2)
    elif similarity_measure == 'chi2alt':
        return utils.chi2alternative_distance(hist1, hist2)
    elif similarity_measure == 'hellinger':
        return utils.hellinger_kernel(hist1, hist2)
    elif similarity_measure == 'all':
        return utils.euclidean_distance(hist1, hist2), utils.histogram_intersection(hist1, hist2), utils.l1_distance(hist1, hist2), utils.chi2_distance(hist1, hist2), utils.chi2alternative_distance(hist1, hist2), utils.hellinger_kernel(hist1, hist2)


# -- CONNECTED COMPONENTS --

def connected_components(mask):
    kernel = np.ones((61,61),np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    mask_closed = np.array(mask_closed, dtype=np.uint8)

    num_comp, components = cv2.connectedComponents(mask_closed)    

    bincount = np.bincount(components.flatten())
    bincount_nonzero = np.delete(bincount,0)
    
    if num_comp == 2:
        index_1 = 1
    else:
        index_1 = np.argmax(bincount_nonzero) + 1
    
        bincount_nonzero = np.delete(bincount_nonzero,index_1 - 1)
        index_2 = np.argmax(bincount_nonzero) + 1
    
        if index_1 <= index_2:
            index_2 = index_2 + 1
    
    
    
    component1 = np.zeros((components.shape[0],components.shape[1]))
    component1[components == index_1] = 1
    
    if num_comp > 2:
        component2 = np.zeros((components.shape[0],components.shape[1]))
        component2[components == index_2] = 1
        
        return [component1,component2]
    
    return [component1]
    

# -- BACKGROUND REMOVAL FUNCTIONS --

def inliers_bounds(u):
    q1 = np.quantile(u, 0.25)  # First quantile
    q3 = np.quantile(u, 0.75)  # Second quantile
    q_inter = q3 - q1  # Interquantile interval

    # Inliers bounds
    upper_bound = q3 + 1.5 * q_inter
    bottom_bound = q1 - 1.5 * q_inter

    return upper_bound, bottom_bound

def inliers(u):
    # Detected border's must be close to each other
    upper_bound, bottom_bound = inliers_bounds(np.extract(u != -1, u))

    # Inliers
    inliers = u
    inliers[u > upper_bound] = -1
    inliers[u < bottom_bound] = -1

    return inliers

def bounds(u):
    i = inliers(u)

    edges = np.argwhere(i != -1)  # Just inliers indexes

    left_i = edges.min()
    left_j = u[left_i]

    right_i = edges.max()
    right_j = u[right_i]

    coordinates = [left_j, left_i, right_j, right_i]

    return coordinates

def last_nonzero(arr, axis, invalid_val=-1):
    flipped_first_nonzero = first_nonzero(np.flip(arr), axis, invalid_val)
    last_n0 = np.flip(flipped_first_nonzero)
    last_n0[last_n0 != -1] = arr.shape[axis] - last_n0[last_n0 != -1]

    return last_n0

def first_nonzero(arr, axis, invalid_val=-1):
    first_n0 = np.where(arr.any(axis=axis), arr.argmax(axis=axis), invalid_val)

    if axis == 0:
        a = arr[first_n0, np.arange(arr.shape[1])]
        first_n0[a == 0] = -1

    elif axis == 1:
        a = arr[np.arange(arr.shape[0]), first_n0]
        first_n0[a == 0] = -1

    return first_n0

def find_mask(connected_component):
    # Find Upper and Bottom borders
    # Takes the first non-zero element's index for each array's column
    upper_border = first_nonzero(connected_component, axis=0, invalid_val=-1)
    bottom_border = last_nonzero(connected_component, axis=0, invalid_val=-1)

    # Find picture's edges coordinates
    if (upper_border > -1).any():
        ul_j,ul_i,ur_j,ur_i = bounds(upper_border)
        bl_j,bl_i,br_j,br_i = bounds(bottom_border)

        pointUL = [ul_i,ul_j] # Upper left point
        pointUR = [ur_i,ur_j] # Upper right point
        pointBL = [bl_i,bl_j] # Bottom left point
        pointBR = [br_i,br_j] # Bottom right point

        # Draw picture's contours
        """ img_contours = cv2.line(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),pointUL,pointUR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5)
        plt.imshow(img_contours)
        plt.show() """

        # Get the mask and convert it to unit8 to not have problems in cv2.CalcHist function later
        mask = cv2.fillConvexPoly(np.zeros((connected_component.shape[0],connected_component.shape[1])),np.array([pointUL,pointUR,pointBR,pointBL]), color=1)
        
        """ mask = np.array(mask, dtype=np.uint8)
        num_labels, labels = cv2.connectedComponents(mask)
        print(num_labels, labels) """
        
        return mask.astype(np.uint8)

        """ plt.imshow(mask, cmap='gray')
        plt.show() """

    else:
        mask = np.zeros((connected_component.shape[0], connected_component.shape[1]), dtype="uint8")
        
        return mask

# -- BACKGROUND REMOVAL EVALUATION FUNCTIONS

def evaluation(predicted, truth):
    tp = np.zeros(predicted.shape)
    fp = np.zeros(predicted.shape)
    fn = np.zeros(predicted.shape)

    tp[(predicted[:, :] == 1) & (truth[:, :] == 1)] = 1
    fp[(predicted[:, :] == 1) & (truth[:, :] == 0)] = 1
    fn[(predicted[:, :] == 0) & (truth[:, :] == 1)] = 1

    """ plt.subplot(221)
    plt.imshow(predicted,cmap='gray')
    plt.subplot(222)
    plt.imshow(tp,cmap='gray')
    plt.subplot(223)
    plt.imshow(fp,cmap='gray')
    plt.subplot(224)
    plt.imshow(fn,cmap='gray')
    plt.show() """

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