import argparse
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from colorDescriptors import colorDescriptors
from text_box import bounding_box
from similarities import hellingerDistance
from evaluation import mapk, bbox_iou
from noise_detection_and_removal import remove_noise


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
    #       2.1. Image retrieval corresponds
    #       2.2. BBox ground truth
    #       2.3. Number of images in the query set
    #       2.4. Number of images in the BBDD set
    with open(args.q / "gt_corresps.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(args.q / "text_boxes.pkl", 'rb') as f:
        boxes = pickle.load(f)

    # Obtain the number of images on the database
    n_query_images = len(glob.glob1(args.q, "*.jpg"))

    # Obtain the number of query images
    n_bbdd_images = len(glob.glob1(args.p, "*.jpg"))

    # -- 3. COMPUTE THE HISTOGRAMS OF THE DATABASE
    bbdd_hists = []
    for i in range(n_bbdd_images):
        img_file = args.p.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
        img = cv2.imread(img_file)
        # Append all the query images histograms
        bbdd_hists.append(colorDescriptors(img, block=args.l))

    # -- 4. ITERATE ON THE QUERY IMAGES
    #       4.1. Obtain the image from the query directory
    #       4.2. Denoise the image
    #       4.3. Compute the BBox of the text
    #       4.4. Compute the histogram without the BBox of the text
    #       4.5. Iterate on the BBDD histograms to compute the Hellinger distance with all the BBDD images
    #       4.6. Sort the best images with best similarities
    #       4.7. Sum the calculation of the IoU between the BBox and its ground truth

    distances = []  # List of distances
    iou = 0         # IoU sum
    for i in range(n_query_images):
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

        query_hist = colorDescriptors(img, block=args.l, mask=mask)

        distances_i = []
        for bbdd_h in bbdd_hists:
            distances_i.append(hellingerDistance(query_hist, bbdd_h))

        arg_distances = np.argsort(distances_i).tolist()
        distances.append(arg_distances[:args.k])
        iou = iou + bbox_iou([left, top, right, bottom], [boxes[i][0][0], boxes[i][0][1], boxes[i][0][2], boxes[i][0][3]])

    # -- 5. DISPLAY THE MAP@K AND mIOU --
    print()
    print(f'mAP@k (k = {args.k}) using HSV block level {args.l} and hellinger distances:')
    print(mapk(data, distances, k=args.k))
    print()
    print('mIoU of the Bounding Boxes:')
    print(iou/(sum([len(b) for b in boxes])))


if __name__ == "__main__":
    main()
