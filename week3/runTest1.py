import argparse
import pickle
import glob
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from colorDescriptors import colorDescriptors
from text_box import bounding_box
from similarities import hellingerDistance
from evaluation import mapk, bbox_iou
from noise_detection_and_removal import remove_noise
from read_text import read_text,extractTextGroundTruth, compareArguments
from LBP_descriptor import LBP_blocks


def parse_args():
    """
    Function to get the input arguments
    Returns
    parse_args()
    """
    parser = argparse.ArgumentParser(description='CBIR with different descriptors and distances')
    parser.add_argument('-k', type=int, default=10,
                        help='Number of images to retrieve')
    parser.add_argument('-l', type=int, default=1,
                        help='Number of levels to compute the histogram')
    parser.add_argument('-p', type=Path, required=True,
                        help='Path to the database directory')
    parser.add_argument('-q', type=Path, required=True,
                        help='Path to the query set directory')
    return parser.parse_args()


def main():

    # -- 1. OBTAIN THE ARGUMENTS --
    args = parse_args()

    # -- 2. OBTAIN CORRESPONDENCES AND GROUND TRUTHS --
    #       2.1. Image retrieval correspondences
    #       2.2. BBox ground truth
    #       2.3. Text retrieval correspondences
    #       2.4. Number of images in the query set
    #       2.5. Number of images in the BBDD set
    text_corresp, text_data = extractTextGroundTruth(args.p)

    # Crete directory in with the txt are stored
    if not os.path.exists('texts'):
        os.makedirs('texts', 0o777)

    # Obtain the number of images on the database
    n_query_images = len(glob.glob1(args.q, "*.jpg"))

    # Obtain the number of query images
    n_bbdd_images = len(glob.glob1(args.p, "*.jpg"))

    # -- 3. COMPUTE THE HISTOGRAMS OF THE DATABASE
    bbdd_hists = []
    for i in tqdm(range(n_bbdd_images)):
        img_file = args.p.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
        img = cv2.imread(img_file)
        # Append all the query images histograms
        color_hist = colorDescriptors(img, block=args.l)
        texture_hist = LBP_blocks(img, p=12, r=2.5, block=args.l)

        bbdd_hists.append(np.concatenate((color_hist, texture_hist)))


    # -- 4. ITERATE ON THE QUERY IMAGES
    #       4.1. Obtain the image from the query directory
    #       4.2. Denoise the image
    #       4.3. Compute the BBox of the text
    #       4.4. Compute the histogram without the BBox of the text
    #       4.5. Iterate on the BBDD histograms to compute the Hellinger distance with all the BBDD images
    #       4.6. Sort the best images with best similarities
    #       4.7. Modify the the best images with the text descriptor
    #       4.8. Sum the calculation of the IoU between the BBox and its ground truth

    bboxes = []
    distances = []  # List of ditances
    iou = 0         # IoU sum
    for i in tqdm(range(n_query_images)):
        img_file = args.q.as_posix() + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
        img = cv2.imread(img_file)

        img = remove_noise(img)

        # Compute the text box
        [left, top, right, bottom] = bounding_box(img)
        mask = np.ones(img.shape[:2], np.uint8)

        # When no BBOX is detected, the bounding box function returns a BBox with the shape of the image
        # So, when this occurs do not substract the BBox from the mask
        if (right != img.shape[0] and right != img.shape[1]):
            mask[top:bottom, left:right] = 0

        color_hist = colorDescriptors(img, block=args.l, mask=mask)
        texture_hist = LBP_blocks(img, p=12, r=2.5, block=args.l, mask=mask)

        query_hist = np.concatenate((color_hist, texture_hist))

        distances_i = []
        for bbdd_h in bbdd_hists:
            distances_i.append(hellingerDistance(query_hist, bbdd_h))

        arg_distances = np.argsort(distances_i).tolist()

        text = read_text(img, [left, top, right, bottom])

        arg_distances = compareArguments(arg_distances, text, text_corresp, text_data)

        distances.append(arg_distances[:args.k])
        bboxes.append([left, top, right, bottom])

        with open('texts/00' + ('00' if i < 10 else '0') + str(i) + '.txt', 'w+') as f:
            f.write(text)

    # -- 5. DISPLAY THE MAP@K AND mIOU --
    with open('result.pkl', 'wb') as f:
        pickle.dump(distances, f)

    with open('text_boxes.pkl', 'wb') as f:
        pickle.dump(bboxes, f)


if __name__ == "__main__":
    main()
