import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt

# -- CHECK ARGUMENTS --

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

# -- IMAGE RETRIEVAL FUNCTIONS --

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