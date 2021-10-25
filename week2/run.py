import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from histograms import computeSimilarity
from mapk import mapk
from utils import checkArguments, get_histograms_from_set


def parse_args():
    """
    Function to get the input arguments
    Returns
    parse_args()
    """
    parser = argparse.ArgumentParser(description='CBIR with different descriptors and distances')
    parser.add_argument('-b', type=str, required=True,
                        help='Remove background from the images (y) or not (n)')
    parser.add_argument('-r', type=int, default='0',
                        help='Define the resolution of the histogram')
    parser.add_argument('-rt', type=str, default='pyramid',
                        help='Define the type: pyramid or level')
    parser.add_argument('-t', type=str, default='n',
                        help='If the images have text (y) or (n)')
    parser.add_argument('-m', type=str, default='d',
                        help='Define if the query set is for development (d) or test(t)')
    parser.add_argument('-k', type=int, default=10,
                        help='Number of images to retrieve')
    parser.add_argument('-c', type=str, required=True,
                        help='Color Space in which the histograms will be computed')
    parser.add_argument('-d', type=str, default='all',
                        help='Distance to compare the histograms')
    parser.add_argument('-p', type=Path, required=True,
                        help='Path to the database directory')
    parser.add_argument('-q', type=Path, required=True,
                        help='Path to the query set directory')
    return parser.parse_args()


def main():
    print()
    # Obtain the arguments
    args = parse_args()

    # Check if the arguments are valid (utils.py)
    checkArguments(args)

    # If the program is in Development, obtain the associated image of the BBDD of each query image
    if args.m == 'd':
        with open(args.q / "gt_corresps.pkl", 'rb') as f:
            data = pickle.load(f)

    # Obtain the number of images on the database
    n = len(glob.glob1(args.p, "*.jpg"))

    # Obtain the number of query images
    t = len(glob.glob1(args.q, "*.jpg"))

    query_hist = get_histograms_from_set(args.q, args)
    BBDD_hist = get_histograms_from_set(args.p, args)

    set_name = str(args.q).split('data')[1]
    mapk_type = 'multiple' if 'qsd1_w2' in set_name else 'single'

    # List of lists
    exp_euclidean = []
    exp_intersection = []
    exp_l1 = []
    exp_chi2 = []
    exp_hellinger = []

    print()
    print('Computing the distances between histograms...')
    for j in tqdm(range(t)):
        # Obtain the histogram of the query image j
        hist = query_hist[j]

        eucl_distances = np.array([])
        instersection_distances = np.array([])
        l1_distances = np.array([])
        chi2_distances = np.array([])
        hellinger_distances = np.array([])

        for i in range(n):

            # Obtain the histogram of the database image i
            db_hist = BBDD_hist[i]

            # Compute the distances
            if args.d == "all":
                eucl_temp, instersection_temp, l1_temp, chi2_temp, hellinger_temp = computeSimilarity(hist, db_hist, similarity_measure='all')
                eucl_distances = np.append(eucl_distances, eucl_temp)
                instersection_distances = np.append(instersection_distances, instersection_temp)
                l1_distances = np.append(l1_distances, l1_temp)
                chi2_distances = np.append(chi2_distances, chi2_temp)
                hellinger_distances = np.append(hellinger_distances, hellinger_temp)

            elif args.d == "euclidean":
                eucl_distances = np.append(eucl_distances, computeSimilarity(hist, db_hist, similarity_measure='euclidean'))

            elif args.d == "intersec":
                instersection_distances = np.append(instersection_distances, computeSimilarity(hist, db_hist, similarity_measure='intersec'))

            elif args.d == "l1":
                l1_distances = np.append(l1_distances, computeSimilarity(hist, db_hist, similarity_measure='l1'))

            elif args.d == "chi2alt":
                chi2alt_distances = np.append(chi2alt_distances, computeSimilarity(hist, db_hist, similarity_measure='chi2alt'))

            elif args.d == "hellinger":
                hellinger_distances = np.append(hellinger_distances, computeSimilarity(hist, db_hist, similarity_measure='hellinger'))

        if args.d == "all":
            exp_euclidean.append(eucl_distances.argsort(axis=0)[:args.k].tolist())
            exp_intersection.append(np.flip(instersection_distances.argsort(axis=0)[-args.k:]).tolist())
            exp_l1.append(l1_distances.argsort(axis=0)[:args.k].tolist())
            exp_chi2.append(chi2_distances.argsort(axis=0)[:args.k].tolist())
            exp_hellinger.append(hellinger_distances.argsort(axis=0)[:args.k].tolist())

        elif args.d == "euclidean":
            exp_euclidean.append(eucl_distances.argsort(axis=0)[:args.k].tolist())

        elif args.d == "intersec":
            exp_intersection.append(np.flip(instersection_distances.argsort(axis=0)[-args.k:]).tolist())

        elif args.d == "chi2":
            exp_chi2.append(chi2_distances.argsort(axis=0)[:args.k].tolist())

        elif args.d == "l1":
            exp_l1.append(l1_distances.argsort(axis=0)[:args.k].tolist())

        elif args.d == "hellinger":
            exp_hellinger.append(hellinger_distances.argsort(axis=0)[:args.k].tolist())

    if args.m == 'd':
        print('mAP@k (K = {}) of the desired distances for {} Color Space and Level {}'.format(int(args.k), str(args.c), str(args.r)))

        if args.d == "all":
            print("Euclidean Distance: {0:.4f}".format(mapk(data, exp_euclidean, args.k,set_name)))
            print("Histogram Intersection: {0:.4f}".format(mapk(data, exp_intersection, args.k,set_name)))
            print("L1 Distance: {0:.4f}".format(mapk(data, exp_l1, args.k,set_name)))
            print("Chi-Squared Distance: {0:.4f}".format(mapk(data, exp_chi2, args.k,set_name)))
            print("Hellinger Distance: {0:.4f}".format(mapk(data, exp_hellinger, args.k,set_name)))

        elif args.d == "euclidean":
            print("Euclidean Distance: {0:.4f}".format(mapk(data, exp_euclidean, args.k,set_name)))

        elif args.d == "intersec":
            print("Histogram Intersection: {0:.4f}".format(mapk(data, exp_intersection, args.k,set_name)))

        elif args.d == "chi2":
            print("Chi-Squared Distance: {0:.4f}".format(mapk(data, exp_chi2, args.k,set_name)))

        elif args.d == "l1":
            print("L1 Distance: {0:.4f}".format(mapk(data, exp_l1, args.k,set_name)))

        elif args.d == "hellinger":
            print("Hellinger Distance: {0:.4f}".format(mapk(data, exp_hellinger, args.k,set_name)))

    elif args.m == 't':
        if args.d == "euclidean":
            with open('result.pkl', 'wb') as f:
                pickle.dump(exp_euclidean, f)
        elif args.d == "intersec":
            with open('result.pkl', 'wb') as f:
                pickle.dump(exp_intersection, f)
        elif args.d == "chi2":
            with open('result.pkl', 'wb') as f:
                pickle.dump(exp_chi2, f)
        elif args.d == "l1":
            with open('result.pkl', 'wb') as f:
                pickle.dump(exp_l1, f)
        elif args.d == "hellinger":
            with open('result.pkl', 'wb') as f:
                pickle.dump(exp_hellinger, f)


if __name__ == "__main__":
    main()