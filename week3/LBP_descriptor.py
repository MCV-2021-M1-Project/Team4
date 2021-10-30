import cv2
import matplotlib.pyplot as plt
from skimage import feature
import numpy as np
from similarities import hellingerDistance

## extract_LBP_features: Compute LBP Histograms of an image
    #   Input:
    #     - image: Input image in BGR, as given by cv2.imread()
    #     - mask: Given a mask, the image is croped to ease computation
    #     - p, r: Number of neighbor points and radius
    #     - type: Specify method of LBP (see https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern)
    #     - p, r: Number of neighbor points and radius
def extract_LBP_features(image, p , r, mask = None, type = 'default', color = 'gray'):
    # plt.subplot(131)
    # plt.imshow(image)
    # plt.subplot(132)
    # plt.imshow(mask)
    cropped = crop_image(image, mask)

    # plt.subplot(133)
    # plt.imshow(cropped)
    # plt.show()

    # Change the color space to use (Gray is better)
    if color == 'gray':
        im = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    elif color == 'HSV':
        im = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        im = im[:, :, 2]

    # compute LBP
    lbp = feature.local_binary_pattern(im, p, r, method=type)

    # Transform to cv2 like
    lbp_cv2 = np.float32(lbp)

    # Compute max value in the histogram to set range
    max_val = int(lbp_cv2.max(axis = 0).max(axis=0))

    # Compute hist and normalize
    hist = cv2.calcHist([np.float32(lbp)], [0], None, [p+2], [0, max_val])
    hist = cv2.normalize(hist, hist).squeeze()
    return hist

def LBPBlockHistogram(image, p=8, r=1, type = 'default', color='gray', mask=None):

    # If background has been applied to the image, compute the centroids of the mask
    if mask is not None:
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    # If the image is from the BBDD set (in other words, the image is cropped right in the frame),
    # find the center of the image
    else:
        # h: height, w: width
        h, w, _ = image.shape
        cX = int(w / 2)
        cY = int(h / 2)

    # List in which all the submasks are stored
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
        # plt.subplot(121)
        # plt.imshow(image)
        # plt.subplot(122)
        # plt.imshow(masks[i])
        # plt.show()
        blockHists = np.concatenate((blockHists, extract_LBP_features(image, p, r, mask=masks[i], type = type, color=color)))

    return blockHists, masks

def LBP_blocks(image, block=1, p=8, r=1, type = 'default', color='gray', mask = None):
    """
        Compute the color (HSV) descriptors of the image. The function has the possibility of computing the histogram
        of the entire image or doing it by blocks of 4 (block==2) and 16 (block==3)
        :param image: image in BGR
        :param block: level of resolution. 1- Entire image. 2- 4 blocks. 3- 16 blocks.
        :param mask: mask of the image if the backgorund ot text boxes have been substracted
        :return: histograms concatenated
        """
    # If block == 1 compute and return the histogram of the entire image
    if block == 1:
        return np.float32(extract_LBP_features(image, p, r, mask=mask, type = type, color=color))

    # If block greater or equal than 2, compute the first level of block histograms
    elif block >= 2:
        histogram, masks = LBPBlockHistogram(image, p =p, r=r, type = type, color=color, mask=mask)

        #  If block == 2 return the histograms of level 2
        if block == 2:
            return np.float32(histogram)

        # If block == 3, compute and return the 16 histograms concatenated from the level 2 masks.
        else:
            histograms = np.array([])
            for submask in masks:
                blockhists, subsubmasks = LBPBlockHistogram(image, p = p, r=r, type = type, color=color, mask=submask)
                histograms = np.concatenate((histograms, blockhists))

            return np.float32(histograms)

def crop_image(image, mask):
    if mask is None:
        return image

    result = cv2.bitwise_and(image, image, mask=mask)
    [_, _, stats, _] = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    m = float('inf')
    for it in stats:
        if it[4] < m:
            smallestCC = it

    # plt.subplot(121)
    # plt.imshow(result)
    # plt.subplot(122)
    #
    # plt.imshow(result[smallestCC[0]:smallestCC[3], smallestCC[1]:smallestCC[2]])
    # plt.show()
    return result[smallestCC[1]: smallestCC[1] + smallestCC[3], smallestCC[0]: smallestCC[0] + smallestCC[2], :]

if __name__ == "__main__":
    path_image = 'C:/Users/Joan/Desktop/Master_Computer_Vision_2021/M1/data/qsd1_w2/00000.jpg'
    p = 8
    r = 1

    img = cv2.imread(path_image)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(221)
    plt.imshow(img)
    hist = extract_LBP_features(img, p, r, mask=None, plot_resuts=False)
    plt.subplot(222)
    plt.plot(hist)

    path_query = 'C:/Users/Joan/Desktop/Master_Computer_Vision_2021/M1/data/BBDD/bbdd_00077.jpg'
    img2 = cv2.imread(path_query)
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    hist2 = extract_LBP_features(img2, p, r, mask=None, plot_resuts=False)
    plt.subplot(224)
    plt.plot(hist2)
    plt.show()

    print(hellingerDistance(hist, hist2))








