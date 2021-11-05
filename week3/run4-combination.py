import argparse
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from combine_descriptors import combinedDescriptors
from pathlib import Path
from tqdm import tqdm
from text_box import bounding_box
from similarities import hellingerDistance, euclidean_distance, l1_distance, chi2_distance, histogram_intersection
from evaluation import mapk, bbox_iou
from noise_detection_and_removal import remove_noise
import os
from read_text import read_text,extractTextGroundTruth, compareArguments

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

    # LBP Args
    parser.add_argument('-pe', type=int, required=True,
                        help='Number of points LBP')
    parser.add_argument('-r', type=float, required=True,
                        help='Radius LBP')
    # Combination args
    parser.add_argument('-color', type=str, required=False,default=None,
                        help='Color Descriptor')
    parser.add_argument('-texture', type=str, required=False, default=None,
                        help='Texture Descriptor')
    parser.add_argument('-text', type=str, required=False, default=None,
                        help='Text Descriptor')

    # Noise removal
    parser.add_argument('-denoise', type=bool, required=False, default=True,
                        help='Noise Removal')

    return parser.parse_args()

# RUN FOR TASK 4-1 - LBP HIST AND COLOR ON QSD1-W3 (NOISE)
def main():
    # Obtain the arguments
    args = parse_args()

    # Open GT files
    with open(args.q / "gt_corresps.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(args.q / "text_boxes.pkl", 'rb') as f:
        boxes = pickle.load(f)

    #Read Ground Truth Text
    text_corresp, text_data = extractTextGroundTruth(args.p)

    # Obtain the number of images on the database
    n_query_images = len(glob.glob1(args.q, "*.jpg"))

    # Obtain the number of query images
    n_bbdd_images = len(glob.glob1(args.p, "*.jpg"))

    # Params to store output and features
    p = args.pe # arg p is chosen
    r = args.r
    b = args.l

    # Empirically set for LBP
    type = 'default'
    color = 'gray'

    # Combination descriptors arguments
    color_descriptor = args.color
    texture_descriptor = args.texture
    text_descriptor = args.text

    # Linea larguisima para calcular el nombre del directorio donde guardar los descriptores
    dirname = '/Descriptors'
    data_path = str(args.q).split('data')[0] + 'data'
    set_name = str(args.q).split('data')[1]
    features_path = data_path + dirname
    histogram_list_pkl = features_path + f'/BBDD_{color_descriptor+"_" if color_descriptor is not None else ""}{texture_descriptor+"_" if texture_descriptor is not None else ""}{str(p) + "_" + str(r).split(".")[0] + "_" if texture_descriptor == "LBP" else "" }b_{str(b)}.pkl'

    # FIRST: compute the histograms of the database
    # Create features directory if not exist
    if (not os.path.exists(features_path)):
        os.mkdir(features_path)

    # Load bbdd features or compute them in case not exist
    if (os.path.exists(histogram_list_pkl)):
        print(f'FIRST : Loading combined histogram list from BBDD: p = {str(p)}, r = {str(r)}, method = {type}, Color Space: {color}, Blocks: {str(b)}')
        with open(histogram_list_pkl, 'rb') as f:
            bbdd_hists = pickle.load(f)
    else:
        print()
        print(f'FIRST: compute LBP histogram list from BBDD: p = {str(p)}, r = {str(r)}, method = {type}, Color Space: {color}, Blocks: {str(b)}')
        bbdd_hists = []

        for i in tqdm(range(n_bbdd_images)):
            img_file = args.p.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
            img = cv2.imread(img_file)

            # Combine the desired descriptors of color and text
            bbdd_hists.append(combinedDescriptors(img, args))

        # Store list in pkl
        with open(histogram_list_pkl, 'wb') as f:
            pickle.dump(bbdd_hists, f)

    # SECOND: Iterate on the query images to:
    #   - Compute the BBox of the text
    #   - Compute the histogram without the BBox of the text
    #   - Iterate on the BBDD histograms to compute the Hellinger distance with all the BBDD images
    #   - Sort the best images with best similarities
    #   - Sum the calculation of the IoU between the BBox and its ground truth
    print()
    print('SECOND: Iterate on the query images:')

    # List of lists
    dist_euclidean = []
    dist_hellinger = []

    iou = 0
    for i in tqdm(range(n_query_images)):

        img_file = args.q.as_posix() + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'

        ## Pocess Query images
        img = cv2.imread(img_file)

        # Denoise image
        if args.denoise == True:
            img = remove_noise(img)

        #print(i, img_file)
        # plt.imshow(img)
        # plt.show()
        #img = remove_noise(img)
        # plt.imshow(img)
        # plt.show()

        # Detect texts
        # Compute the text box
        [left, top, right, bottom] = bounding_box(img)
        mask = np.ones(img.shape[:2], np.uint8)

        # When no BBOX is detected, the bounding box function returns a BBox with the shape of the image
        # So, when this occurs do not substract the BBox from the mask
        if (right != img.shape[0] and right != img.shape[1]):
            mask[top:bottom, left:right] = 0

        #query_hist = extract_LBP_features(img, p, r, mask=mask, type = type, color=color)
        query_hist = combinedDescriptors(img, args)

        # plt.plot(query_hist)
        # plt.show()

        dist_euclidean_i = []
        dist_hellinger_i = []

        for bbdd_h in bbdd_hists:
            if not texture_descriptor=='DCT':
                dist_euclidean_i.append(euclidean_distance(query_hist, bbdd_h))
                dist_hellinger_i.append(hellingerDistance(query_hist, bbdd_h))
            else:
                # In the case of DCT we compute distances separately

                # First the color at [0]
                color_euclidean = euclidean_distance(query_hist[0], bbdd_h[0])
                color_hellinger = hellingerDistance(query_hist[0], bbdd_h[0])

                # Second the texture at [1]
                texture_similarity = euclidean_distance(query_hist[1], bbdd_h[1])

                dist_euclidean_i.append(color_euclidean + texture_similarity)
                dist_hellinger_i.append(color_hellinger + texture_similarity)


        sort_dist_euclidean = np.argsort(dist_euclidean_i).tolist()
        sort_dist_hellinger = np.argsort(dist_hellinger_i).tolist()

        if args.text is not None:
            text = read_text(img, [left, top, right, bottom])

            sort_dist_euclidean = compareArguments(sort_dist_euclidean, text, text_corresp, text_data)
            sort_dist_hellinger = compareArguments(sort_dist_hellinger, text, text_corresp, text_data)

        dist_euclidean.append(sort_dist_euclidean[:args.k])
        dist_hellinger.append(sort_dist_hellinger[:args.k])

        # Evalate boundingbox
        # if 'w3' in set_name:
        #     bbox_gt = [boxes[i][0][0], boxes[i][0][1], boxes[i][0][2], boxes[i][0][3]]
        # else:
        #     bbox_gt = [boxes[i][0][0][0], boxes[i][0][0][1], boxes[i][0][2][0], boxes[i][0][2][1]]
        #
        # iou = iou + bbox_iou([left, top, right, bottom], bbox_gt)

    # Compute and print the mAP and mIoU
    print()
    print(f'mAP@k (k = {args.k}) using p = {str(p)}, r = {str(r)}, method = {type}, Color Space: {color}, Blocks: {str(b)}')
    print()
    print(f'Euclidean: {mapk(data, dist_euclidean, k=args.k)}')
    print(f'Hellinger: {mapk(data, dist_hellinger, k=args.k)}')
    print()
    print('mean IoU of the Bounding Boxes:')
    print(iou/n_query_images)



if __name__ == "__main__":
    main()
