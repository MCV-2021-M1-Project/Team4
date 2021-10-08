import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
## -- SIMILARITY MEASURES --
def euclidean_distance(u,v):
    return np.linalg.norm(u - v)

def l1_distance(u,v):
    return np.linalg.norm((u - v),ord=1)

def chi2_distance(u,v):
    return np.sum(np.nan_to_num(np.divide(np.power(u-v,2),(u+v))))

def histogram_intersection(u,v):
    return np.sum(np.minimum(u,v))

def hellinger_kernel(u,v):
    return np.sum(np.sqrt(np.multiply(u,v)))

def computeHistImage(image, color_space):
    if color_space == "GRAY":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = [0]
    elif color_space == "RGB":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Already BGR
        channels = [0, 1, 2]
    elif color_space == "HSV":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [0, 1]
    elif color_space == "YCrCb":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = [0, 1, 2]

    # Compute hist
    image_hist = np.empty([0, 1])

    for c in channels:
        channel_hist = cv2.calcHist([image_color], [c], None, [256], [0, 256])
        cv2.normalize(channel_hist, channel_hist)
        image_hist = np.concatenate((image_hist, channel_hist), axis=0)

    # plt.plot(image_hist)
    # plt.show()

    return image_hist

def computeSimilarity(hist1, hist2, similarity_measure):
    if similarity_measure == 'euclidean':
        res = utils.euclidean_distance(hist1, hist2)
    elif similarity_measure == 'hist_intersec':
        res = utils.histogram_intersection(hist1, hist2)
    elif similarity_measure == 'chi2':
        res = utils.chi2_distance(hist1, hist2)
    elif similarity_measure == 'hellinger':
        res = utils.hellinger_kernel(hist1, hist2)

    return res