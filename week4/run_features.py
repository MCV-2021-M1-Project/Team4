import argparse
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from text_box import bounding_box
from evaluation import bbox_iou, mapk2paintings
from noise_detection_and_removal import remove_noise
from feature_descriptors import feature_descriptor, check_painting_db
from similarities import feature_distance
from read_text import extractTextGroundTruth, compareArguments, read_text
from background_substraction import substractBackground


def parse_args():
    """
    Function to get the input arguments
    Returns
    parse_args()
    """
    parser = argparse.ArgumentParser(description='CBIR with different descriptors and distances')
    parser.add_argument('-k', type=int, default=10,
                        help='Number of images to retrieve')
    parser.add_argument('-f', type=str, required=True,
                        help='Feature')
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
    #       2.3. Text retrieval correspondences
    #       2.3. Number of images in the query set
    #       2.4. Number of images in the BBDD set

    with open(args.q / "gt_corresps.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(args.q / "text_boxes.pkl", 'rb') as f:
        boxes = pickle.load(f)

    text_corresp, text_data = extractTextGroundTruth(args.p)

    # Obtain the number of images on the database
    n_query_images = len(glob.glob1(args.q, "*.jpg"))

    # Obtain the number of query images
    n_bbdd_images = len(glob.glob1(args.p, "*.jpg"))

    # -- 3. COMPUTE/LOAD THE HISTOGRAMS OF THE DATABASE

    bbdd_hists = []
    print("BBDD:")
    for i in tqdm(range(n_bbdd_images)): # range(n_bbdd_images)
        img_file = args.p.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
        img = cv2.imread(img_file)
        # Append all the query images histograms
        bbdd_hists.append(feature_descriptor(img, mask=None, type=args.f))

    with open('bbdd_features.pkl', 'wb') as f:
        pickle.dump(bbdd_hists, f)
    """
    with open('bbdd_features.pkl', 'rb') as f:
        bbdd_hists = pickle.load(f)
    """
    # -- 4. ITERATE ON THE QUERY IMAGES
    #       4.1. Obtain the image from the query directory
    #       4.2. Denoise the image
    #       4.3. Substract the background
    #       4.4. Iterate on every painting mask
    #           4.4.1. Compute the BBox of the text for each mask
    #           4.4.2. Compute the histogram without the BBox of the text
    #           4.4.3. Iterate on the BBDD histograms to compute the distance with all the BBDD images
    #           4.4.4. Sort the best images with best similarities
    #           4.4.5. Modify the the best images with the text descriptor
    #           4.4.6. Sum the calculation of the IoU between the BBox and its ground truth

    distances = []  # List of distances
    bboxes = []     # List of bboxes
    iou = 0         # IoU sum
    print("Querys y comparaciones:")
    for i in tqdm(range(n_query_images)): # range(n_query_images)
        img_file = args.q.as_posix() + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
        img = cv2.imread(img_file)

        img = remove_noise(img)

        # Obtain the backgorund masks of the image. masks is a list of masks. If there is only a painting in the image
        # the length will be 1, 2 paintings 2, and 3 paintings 3.
        masks = substractBackground(img)

        image_distances = []
        image_bboxes = []
        for idx, mask in enumerate(masks):
            # Compute the text box
            [left, top, right, bottom] = bounding_box(img, mask=mask)

            """
            img_to_show = cv2.hconcat([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask*255])
            cv2.imshow('image' + str(i) + '_' + str(idx), img_to_show)
            cv2.waitKey(5000)
            """

            # When no BBOX is detected, the bounding box function returns a BBox with the shape of the image
            # So, when this occurs do not substract the BBox from the mask
            if right != img.shape[0] and right != img.shape[1]:
                mask[top:bottom, left:right] = 0

            query_hist = feature_descriptor(img, mask=mask, type=args.f)

            image_bboxes.append([left, top, right, bottom])
            distances_i = []
            for bbdd_h in bbdd_hists:
                if bbdd_h is None or query_hist is None:
                    distances_i.append(10000000)
                else:
                    distances_i.append(feature_distance(query_hist, bbdd_h, type='sift_method2'))

            iou = iou + bbox_iou([left, top, right, bottom], boxes[i][idx])

            text = read_text(img, [left, top, right, bottom])

            arg_distances = np.argsort(distances_i).tolist()
            arg_distances = compareArguments(arg_distances, text, text_corresp, text_data)
            image_distances.append(arg_distances[:args.k])

            # Add a - 1 if the image is not in the bg:
            image_distances[idx] = check_painting_db(distances_i, image_distances[idx], th=0.24) #sift 0.03, orb 0.24


        distances.append(image_distances)
        bboxes.append(image_bboxes)

    # -- 5. DISPLAY THE MAP@K --
    print('a')
    print()
    print(f'mAP@k (k = {args.k}):')
    print(mapk2paintings(data, distances, k=args.k))

if __name__ == "__main__":
    main()
