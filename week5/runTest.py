import argparse
import os.path
import pickle
import glob
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage

from text_box import bounding_box
from evaluation import bbox_iou, mapk2paintings, angular_error
from noise_detection_and_removal import remove_noise
from feature_descriptors import feature_descriptor, check_painting_db
from similarities import feature_distance
#from read_text import extractTextGroundTruth, compareArguments, read_text
from background_substraction import substractBackground
from read_text import extractTextGroundTruth, read_text, compareArguments
from rotation import rotate_image, find_frame_points

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
    #       2.1. Text retrieval correspondences
    #       2.2. Number of images in the query set
    #       2.3. Number of images in the BBDD set
    #       2.4. If the database has a pickle file with de descriptors load it, if not create it

    text_corresp, text_data = extractTextGroundTruth(args.p)

    # Crete directory in with the txt are stored
    if not os.path.exists('texts'):
        os.makedirs('texts', 0o777)

    # Obtain the number of images on the database
    n_query_images = len(glob.glob1(args.q, "*.jpg"))

    # Obtain the number of query images
    n_bbdd_images = len(glob.glob1(args.p, "*.jpg"))

    file_name = args.f + 'features.pkl'

    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            bbdd_hists = pickle.load(f)

    else:
        bbdd_hists = []
        for i in tqdm(range(n_bbdd_images)):  # range(n_bbdd_images)
            img_file = args.p.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
            img = cv2.imread(img_file)
            # Append all the query images histograms
            bbdd_hists.append(feature_descriptor(img, mask=None, type=args.f))

        with open(file_name, 'wb') as f:
            pickle.dump(bbdd_hists, f)

    distances = []  # List of distances
    bboxes = []     # List of bboxes
    for i in tqdm(range(n_query_images)): # range(n_query_images)

        img_file = args.q.as_posix() + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
        img = cv2.imread(img_file)

        img = remove_noise(img)

        img, angle = rotate_image(img)

        # Transform angle from [-90, 90] to [0, 180]
        predicted_angle = 0
        if angle > 0:
            predicted_angle = 180 - angle
        elif angle < 0:
            predicted_angle = -angle

        # Obtain the backgorund masks of the image. masks is a list of masks. If there is only a painting in the image
        # the length will be 1, 2 paintings 2, and 3 paintings 3.
        masks = substractBackground(img, plot=False)
        image_distances = []
        image_bboxes = []
        text_from_image = []
        for idx, mask in enumerate(masks):
            # Prepare the pickle file: frames.pkl
            image_bboxes.append([predicted_angle, find_frame_points(mask, angle)])

            # Compute the text box
            [left, top, right, bottom] = bounding_box(img, mask=mask)

            # When no BBOX is detected, the bounding box function returns a BBox with the shape of the image
            # So, when this occurs do not substract the BBox from the mask
            if bottom != img.shape[0] and right != img.shape[1]:
                mask[top:bottom, left:right] = 0

            query_hist = feature_descriptor(img, mask=mask, type=args.f)

            distances_i = []
            for bbdd_h in bbdd_hists:
                if bbdd_h is None or query_hist is None:
                    distances_i.append(10000000)
                else:
                    distances_i.append(feature_distance(query_hist, bbdd_h, type='orb'))

            text = read_text(img, [left, top, right, bottom])

            arg_distances = np.argsort(distances_i).tolist()
            arg_distances, predictedText = compareArguments(arg_distances, text, text_corresp, text_data)
            text_from_image.append(predictedText)
            image_distances.append(arg_distances[:args.k])

            # Add a - 1 if the image is not in the bg:
            image_distances[idx] = check_painting_db(distances_i, image_distances[idx], type=args.f)  # sift 0.03, orb 0.19

        distances.append(image_distances)
        bboxes.append(image_bboxes)

        with open('texts/00' + ('00' if i < 10 else '0') + str(i) + '.txt', 'w+') as f:
            for line in text_from_image:
                f.write(line)
                f.write('\n')

    # -- 5. SAVE PICKLE FILES --
    with open('result.pkl', 'wb') as f:
        pickle.dump(distances, f)

    with open('frames.pkl', 'wb') as f:
        pickle.dump(bboxes, f)

if __name__ == "__main__":
    main()