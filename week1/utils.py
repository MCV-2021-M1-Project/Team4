import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt


# -- SIMILARITY MEASURES --
def euclidean_distance(u,v):
    return np.linalg.norm(u - v)


def l1_distance(u,v):
    return np.linalg.norm((u - v),ord=1)


def chi2_distance(u,v, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(u, v)])
    #return np.sum(np.nan_to_num(np.divide(np.power(u-v,2),(u+v))))


def histogram_intersection(u,v):
    return np.sum(np.minimum(u,v))


def hellinger_kernel(u,v):
    # return np.sum(np.sqrt(np.multiply(u,v)))
    n = len(u)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(u[i]) - np.sqrt(v[i])) ** 2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result


def checkArguments(args):
    if args.m != 'd' and args.m != 't':
        raise TypeError("The modes of the code are: development(d) and test(t)")
    if args.c not in ["GRAY", "RGB", "H", "S", "V", "HS", "HV", "HSV", "YCrCb", "CrCb", "CIELab"]:
        raise TypeError("Wrong color space")
    if args.d not in ["all", "euclidean", "intersec", "l1", "chi2", "hellinger"]:
        raise TypeError("Wrong distance")
    if args.m == 't' and args.d == 'all':
        raise Exception("The test mode cannot be done with all the distances at the same time given that only one "
                        "distance can be stored in the same file")


def computeHistImage(image, color_space):
    if color_space == "GRAY":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = [0]
    elif color_space == "RGB":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Already BGR
        channels = [0, 1, 2]
    elif color_space == "H":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [0]
    elif color_space == "S":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [1]
    elif color_space == "V":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [2]
    elif color_space == "HS":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [0, 1]
    elif color_space == "HV":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [0, 2]
    elif color_space == "HSV":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = [0, 1, 2]
    elif color_space == "YCrCb":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = [0, 1, 2]
    elif color_space == "CrCb":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = [1, 2]
    elif color_space == "CIELab":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
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
        return utils.euclidean_distance(hist1, hist2)
    elif similarity_measure == 'intersec':
        return utils.histogram_intersection(hist1, hist2)
    elif similarity_measure == 'l1':
        return utils.l1_distance(hist1, hist2)
    elif similarity_measure == 'chi2':
        return utils.chi2_distance(hist1, hist2)
    elif similarity_measure == 'hellinger':
        return utils.hellinger_kernel(hist1, hist2)
    elif similarity_measure == 'all':
        return utils.euclidean_distance(hist1, hist2), utils.histogram_intersection(hist1, hist2), utils.l1_distance(hist1, hist2), utils.chi2_distance(hist1, hist2), utils.hellinger_kernel(hist1, hist2)

