import cv2
import numpy as np
from histograms import computeHistImage
import matplotlib.pyplot as plt


def blockHistogram(image, color_space, mask=None):

    # If background has been applied to the image, compute the centroids of the mask
    print(type(mask))
    """ plt.imshow(mask)
    plt.show() """
    if mask is not None:
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    # If the image is from the BBDD set (in other words, the image is cropped right in the frame),
    # find the center of the image
    else:
        # h: height, w: width, c: channels(not used)
        h, w, c = image.shape
        cX = int(w / 2)
        cY = int(h / 2)
    # List in which all the masks are stored
    masks = []
    pos = 0

    # From the centroid (either from the BBDD image or from the mask) divide the image in 4 different masks
    for i in range(2):
        for j in range(2):
            masks.append(np.zeros((image.shape[0], image.shape[1]), dtype="uint8"))
            masks[pos][cY * i: cY * (i + 1), cX * j: cX * (j + 1)] = 1
            pos = pos + 1

    # Compute the intersection of the original mask and all the multiresolution masks
    if mask is not None:
        for i in range(len(masks)):
            masks[i] = masks[i] * mask

    blockHists = np.array([])
    for i in range(len(masks)):
        blockHists = np.concatenate((blockHists, computeHistImage(image, color_space=color_space, mask=masks[i])))

    return blockHists, masks


def multiresolution(image, color_space, level, type, mask=None):
    """ plt.imshow(image)
    plt.show() """
    if type == 'pyramid':
        # Compute the histogram of the original whole image and store it in a variable in which all the
        # histograms will be concatenated
        histograms = np.concatenate(computeHistImage(image, color_space=color_space, mask=mask)[:,np.newaxis])

        # Compute the 2nd level histograms (4 blocks) and concatenate them to the histogram variable
        if level >= 2:
            blockHistograms, masks = blockHistogram(image, color_space, mask=mask)
            histograms = np.concatenate((histograms, blockHistograms))

        # Compute the 3rd level histograms (16 blocks) and concatenate them to the histogram variable
        if level == 3:
            for i in range(len(masks)):
                blockHistograms, block_masks = blockHistogram(image, color_space, mask=masks[i])
                histograms = np.concatenate((histograms, blockHistograms))
    else:
        # Compute the 2nd level histograms (4 blocks) and concatenate them to the histogram variable
        if level > 1:
            blockHistograms2, masks2 = blockHistogram(image, color_space, mask=mask) # Level 2
            if level == 2:
                histograms = blockHistograms2
            else:
                histograms = np.array([])
                for i in range(len(masks2)):
                    blockHistograms3, block_masks = blockHistogram(image, color_space, mask=masks2[i])
                    histograms = np.concatenate((histograms, blockHistograms3))

    return histograms

"""
image = cv2.imread('/home/david/Desktop/M1/data/BBDD/bbdd_00015.jpg')

#masks = substractBackground(numImages=30, query_path=Path('/home/david/Desktop/M1/data/qsd2_w1/'), mode='d')

# mask = masks[1]


histograms_bbdd = multiresolution(image=image, color_space='GRAY', level=3, mask=None)
#histograms_query = multiresolution(image=image, color_space='GRAY', level=3, mask=mask)

print(histograms_query.shape, histograms_bbdd.shape)
"""

    
    
    
