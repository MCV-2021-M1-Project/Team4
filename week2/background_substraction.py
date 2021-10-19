import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from PIL import Image
from tqdm import tqdm
from utils import first_nonzero, last_nonzero, bounds, evaluation, connected_components, find_mask

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
    masks = []
    evaluations = []
    for j in tqdm(range(numImages)):
        img_file = args.q + '/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
        img = cv2.imread(img_file)

        # RGB to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Saturation and Value channels
        s = hsv_img[:, :, 1]
        v = hsv_img[:, :, 2]

        # Saturation and Value thresholding
        thresholded = np.zeros((img.shape[0], img.shape[1]))
        thresholded[(s > TH_S) | (v < TH_V)] = 1

        # Find the two biggest connected components
        components = connected_components(thresholded)
        
        # Compute connected components' masks
        mask = np.zeros((img.shape[0], img.shape[1]),dtype="uint8")
        
        for cc in range(len(components)):
            mask = mask | find_mask(components[cc])

        masks.append(mask)

        if args.m == 'd':
            # Evaluations
            ground_truth_file = args.q + '/00' + ('00' if j < 10 else '0') + str(j) + '.png'
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

    if args.m == 'd':

        evaluation_mean = np.sum(evaluations, axis=0) / numImages
        print()
        print("BACKGROUND SUBSTRACTION MEASURES:")
        print("Precision: {0:.4f}".format(evaluation_mean[0]))
        print("Recall: {0:.4f}".format(evaluation_mean[1]))
        print("F1-measure: {0:.4f}".format(evaluation_mean[2]))

    return masks
