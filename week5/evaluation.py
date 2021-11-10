import numpy as np

# FILE WITH ALL THE FUNCTIONS WHICH DEAL WITH EVALUATING THE CODE

# -- RETRIEVAL EVALUATION --

def apk(actual, predicted, k=10):

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def mapk2paintings(actual, predicted, k=10):
    """
    mapk for 2 paintings
    """

    x = []
    for it in predicted:
        for a in it:
            x.append(a)

    y = []
    for it in actual:
        for a in it:
            li = [a]
            y.append(li)

    return np.mean([apk(a,p,k) for a,p in zip(y, x)])

# -- BACKGROUND SUBTRACTION EVALUATION --

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

# -- TEXT BOX EVALUATION --

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou


