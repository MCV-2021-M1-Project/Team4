import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

TH_S = 114  # Saturation threshold
TH_V = 63   # Value threshold


def substractBackground(numImages, args):
    """
    Function to substract the background from the images
    Parameters
    ----------
    numImages: number of images of the query set
    args: input parameters
    Returns: List of images with the background removed. The images are flat (1 empty dimension), their shape is lost
    -------
    """
    print('Estimating and substracting the background for every query image...')
    set_name = str(args.q).split('data')[1]
    overall_textboxes = []
    masks = []
    evaluations = []
    for j in tqdm(range(numImages)):

        img_file = args.q.as_posix() + '/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'

        img = cv2.imread(img_file)
        """ print(img_file) """
        # plt.imshow(img)
        # plt.show()
        # RGB to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Saturation and Value channels
        s = hsv_img[:, :, 1]
        v = hsv_img[:, :, 2]

        # Saturation and Value thresholding
        thresholded = np.zeros((img.shape[0], img.shape[1]))
        thresholded[(s > TH_S) | (v < TH_V)] = 1

        """ plt.subplot(221)
        plt.imshow(thresholded,cmap="gray") """

        # Find the two biggest connected components
        # If 2 paintings it outputs 2 components
        components = connected_components(thresholded,set_name)

        # Compute connected components' masks
        mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
        text_boxes = []
        every_mask = []
        overlapping = np.ones((img.shape[0], img.shape[1]), dtype="uint8")
        for cc in range(len(components)):
            mask_cc = find_mask(components[cc])
            overlapping = overlapping * mask_cc
            every_mask.append(mask_cc)


        if set_name == '/qsd2_w2' or set_name == '/qst2_w2':
            if overlapping.any() == 1:
                mask_size = 0
                for e in every_mask:
                    aux = np.sum(e)

                    if mask_size < aux:
                        mask_size = aux
                        mask = e

                if set_name == '/qsd1_w2' or set_name == '/qsd2_w2' or set_name == '/qst1_w2' or set_name == '/qst2_w2':
                    text_box = bounding_box(v, mask)
                    
                    
                    point1 = [text_box[0],text_box[1]]
                    point2 = [text_box[2],text_box[1]]
                    point3 = [text_box[0],text_box[3]]
                    point4 = [text_box[2],text_box[3]]
                    
                    """ [pointUL, pointUR, pointBR, pointBL] """
                    
                    text_mask = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([point1,point2,point4,point3]), color=1)
                    mask = mask | text_mask.astype(np.uint8)
        
                    text_boxes.append(text_box)
                
                masks.append([mask])


            else:
                for e in every_mask:
                    # print( e.flatten().sum())
                    # if e.flatten().sum() <  5000:
                    #     continue
                    mask = mask | find_mask(e)
                    if set_name == '/qsd1_w2' or set_name == '/qsd2_w2' or set_name == '/qst1_w2' or set_name == '/qst2_w2':
                        text_box = bounding_box(v, e)
                        text_boxes.append(text_box)
                        point1 = [text_box[0],text_box[1]]
                        point2 = [text_box[2],text_box[1]]
                        point3 = [text_box[0],text_box[3]]
                        point4 = [text_box[2],text_box[3]]
                        
                        text_mask = cv2.fillConvexPoly(np.zeros((img.shape[0],img.shape[1])),np.array([point1,point2,point4,point3]), color=1)
                        mask = mask | text_mask.astype(np.uint8)
                        
                    
                masks.append([mask])
                        
        else:
            for e in every_mask:
                # print( e.flatten().sum())
                # if e.flatten().sum() <  5000:
                #     continue
                if set_name == '/qsd1_w2' or set_name == '/qst1_w2':
                    e = np.ones((img.shape[0],img.shape[1]))
                mask = mask | find_mask(e)
                
                if set_name != '/BBDD':
                    text_box = bounding_box(v, e)
                    text_boxes.append(text_box)
                
                
                    point1 = [text_box[0],text_box[1]]
                    point2 = [text_box[2],text_box[1]]
                    point3 = [text_box[0],text_box[3]]
                    point4 = [text_box[2],text_box[3]]

                    
                    mask = cv2.fillConvexPoly(mask,np.array([point1,point2,point4,point3]), color=0)
                    """ plt.imshow(mask)
                    plt.show() """
                
            masks.append([mask])
            """ mask = mask | text_mask.astype(np.uint8) """

        overall_textboxes.append(text_boxes)
        
        if set_name == '/qst1_w2' or set_name == '/qst2_w2':
            with open(args.q.as_posix() + 'results.pkl','wb') as f:
                pickle.dump(overall_textboxes, f)

        if args.m == 'd' and '/qsd2_w2' in set_name:
            # Evaluations
            ground_truth_file = args.q.as_posix() + '/00' + ('00' if j < 10 else '0') + str(j) + '.png'
            ground_truth = cv2.imread(ground_truth_file)
            ground_truth[ground_truth == 255] = 1  # Range [0,255] to [0,1]

            # Evaluation
            evaluations.append(evaluation(mask, ground_truth[:, :, 0]))

            # If the query set is test, save all the image masks in a directory called masks
        elif args.m == 't':
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if mask[x, y] == 1:
                        mask[x, y] = 255
            if j == 0:
                if not os.path.exists('masks'):
                    os.makedirs('masks', 0o777)
            mask_file = 'masks/00' + ('00' if j < 10 else '0') + str(j) + '.png'
            cv2.imwrite(mask_file, mask)

    if args.m == 'd' and not ('/qsd1_w2' or '/qst1_w2'):
        evaluation_mean = np.sum(evaluations, axis=0) / numImages
        print()
        print("BACKGROUND SUBSTRACTION MEASURES:")
        print("Precision: {0:.4f}".format(evaluation_mean[0]))
        print("Recall: {0:.4f}".format(evaluation_mean[1]))
        print("F1-measure: {0:.4f}".format(evaluation_mean[2]))

    if set_name == '/qsd1_w2' or set_name == '/qsd2_w2' or set_name == '/qst1_w2' or set_name == '/qst2_w2':
        return masks, overall_textboxes
    
    else:
        return masks


# -- CONNECTED COMPONENTS --
def biggest_cc(mask):
    num_comp, cc = cv2.connectedComponents(mask)
    bincount = np.bincount(cc.flatten())
    bincount = np.delete(bincount, 0)  # Just CC  != 0
    bincount_sorted = np.argsort(bincount)[::-1] + 1
    mask_cc = np.zeros((cc.shape[0], cc.shape[1]))
    
    try:
        mask_cc[cc == bincount_sorted[0]] = 1
    except:
        """ print("error mask") """
        return np.ones((cc.shape[0], cc.shape[1]))
    return mask_cc


def connected_components(mask,set_name):
    kernel = np.ones((63, 63), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_closed = np.array(mask_closed, dtype=np.uint8)

    """ plt.subplot(222)
    plt.imshow(mask_closed,cmap="gray") """

    num_comp, components = cv2.connectedComponents(mask_closed)

    bincount = np.bincount(components.flatten())
    bincount_nonzero = np.delete(bincount, 0)

    if set_name == '/qsd2_w2':
        if num_comp == 2:
            index_1 = 1
        else:
            index_1 = np.argmax(bincount_nonzero) + 1

            bincount_nonzero = np.delete(bincount_nonzero, index_1 - 1)
            index_2 = np.argmax(bincount_nonzero) + 1

            if index_1 <= index_2:
                index_2 = index_2 + 1
                
    else:
        index_1 = 1

    component1 = np.zeros((components.shape[0], components.shape[1]))
    component1[components == index_1] = 1

    if set_name == '/qsd2_w2':
        if num_comp > 2:
            component2 = np.zeros((components.shape[0], components.shape[1]))
            component2[components == index_2] = 1

            # plt.subplot(223)
            # plt.imshow(component1, cmap="gray")
            # plt.subplot(224)
            # plt.imshow(component2, cmap="gray")
            # plt.show()

            return [component1, component2]
    
    """ plt.subplot(223)
    plt.imshow(component1, cmap="gray") """
    # plt.show()
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


def last_nonzero(arr, axis):
    flipped_first_nonzero = first_nonzero(np.flip(arr), axis)
    last_n0 = np.flip(flipped_first_nonzero)
    last_n0[last_n0 != -1] = arr.shape[axis] - last_n0[last_n0 != -1]

    return last_n0


def first_nonzero(arr, axis):
    first_n0 = np.where(arr.any(axis=axis), arr.argmax(axis=axis), -1)

    if axis == 0:
        a = arr[first_n0, np.arange(arr.shape[1])]

    elif axis == 1:
        a = arr[np.arange(arr.shape[0]), first_n0]

    first_n0[a == 0] = -1

    return first_n0


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

        # Draw picture's contours
        """ img_contours = cv2.line(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),pointUL,pointUR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointUR,pointBR, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBR,pointBL, color=255,thickness =5)
        img_contours = cv2.line(img_contours,pointBL,pointUL, color=255,thickness =5)
        plt.imshow(img_contours)
        plt.show() """

        # Get the mask and convert it to unit8 to not have problems in cv2.CalcHist function later
        mask = cv2.fillConvexPoly(np.zeros((connected_component.shape[0], connected_component.shape[1])),
                                  np.array([pointUL, pointUR, pointBR, pointBL]), color=1)

        """ plt.imshow(mask,cmap='gray',vmin=0,vmax=1)
        plt.show() """

        return mask.astype(np.uint8)

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


## BOUNDING BOX

def bounding_box(value, mask=[]):
    # Convert blacks to white
    abs_v = np.absolute(value - np.amax(value) / 2)

    # Blackhat and thresholding
    blackhat = morph_filter(abs_v, (3, 3), cv2.MORPH_BLACKHAT)
    blackhat = blackhat / np.max(blackhat)

    if mask != []:
        blackhat = mask * blackhat

    mask = np.zeros((value.shape[0], value.shape[1]))
    mask[blackhat > 0.4] = 1

    # Morphological filters
    mask = morph_filter(mask, (2, 10), cv2.MORPH_CLOSE)  ##Fill letter
    mask = morph_filter(mask, (4, 4),
                        cv2.MORPH_OPEN)  # Delete vetical kernel = (1,3) and horizontal lines kernel = (4,1). Total kernel = (4,3)
    mask = morph_filter(mask, (1, 29), cv2.MORPH_CLOSE)  # Join letters
    # plt.imshow(mask)
    # plt.show()
    ## Find biggest connected component
    cc = biggest_cc(mask.astype(np.uint8))

    # Find component's rectangle's i coordinates
    coord_i = np.where(np.amax(cc, axis=1))
    top = coord_i[0][0]
    bottom = coord_i[0][-1]

    # Expand coordinates and take original image's values in that zone
    inter = bottom - top
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < value.shape[0] else value.shape[0]

    box = np.zeros((value.shape[0], value.shape[1] + 202))
    mask = np.zeros((box.shape[0], box.shape[1]))

    box[top:bottom, 116:box.shape[1] - 116] = blackhat[top:bottom, 15:value.shape[1] - 15]
    box = box / np.amax(box)
    mask[box > 0.46] = 1

    # Morphological filter
    mask = morph_filter(mask, (5, 14), cv2.MORPH_CLOSE)  # Fill letter
    mask = morph_filter(mask, (4, 4),
                        cv2.MORPH_OPEN)  # Delete vertical lines (kernel = (1,4)) and horizontal lines (kernel = (4,1)). Total kernel = (4,3)
    mask = morph_filter(mask, (1, 91), cv2.MORPH_CLOSE)  # Join letters
    mask = morph_filter(mask, (1, 2), cv2.MORPH_OPEN)  # Delete remaining vertical lines

    ##Find biggest connected component
    cc = biggest_cc(mask.astype(np.uint8))
    
    """ plt.imshow(cc)
    plt.show() """

    # Find component's rectangle's i coordinates
    coord_i = np.where(np.amax(cc[:, 101:-101], axis=1))
    coord_j = np.where(np.amax(cc[:, 101:-101], axis=0))
    top = coord_i[0][0]
    bottom = coord_i[0][-1]
    left = coord_j[0][0]
    right = coord_j[0][-1]

    # Expand coordinates and take original image's values in that zone
    inter = int((bottom - top) * 0.5)
    top = top - inter if top - inter > 0 else 0
    bottom = bottom + inter if bottom + inter < value.shape[0] else value.shape[0]
    left = left - inter if left - inter > 0 else 0
    right = right + inter if right + inter < value.shape[1] else value.shape[1]
    coordinates = [left, top, right, bottom]

    """ box = np.zeros((value.shape[0], value.shape[1]))
    box[top:bottom,left:right] = 1

    plt.imshow(box,cmap='gray')
    plt.show() """
    """ text_boxes = data[j][0]
    ground_truth = cv2.fillConvexPoly(np.zeros((value.shape[0],value.shape[1])),np.array([text_boxes[0],text_boxes[1],text_boxes[2],text_boxes[3]]), color=1)
    # Evaluation
    ev = evaluation(box,ground_truth)
    evaluations.append(ev) """

    return coordinates


##MORPHOLOGICAL FILTERS

def morph_filter(mask, kernel, filter):
    return cv2.morphologyEx(mask, filter, np.ones(kernel, np.uint8))