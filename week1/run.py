import argparse
import pickle
import cv2
import os
import glob
import numpy as np
import pathlib
from tqdm import tqdm
from pathlib import Path
from mapk import mapk
from utils import computeHistImage, computeSimilarity, checkArguments


def parse_args():
    parser = argparse.ArgumentParser(description='CBIR with different descriptors and distances')
    parser.add_argument('-m', type=str, default='d',
                        help='Define if the query set is for development (d) or test(t).')
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

    # Check if the arguments are valid
    checkArguments(args)

    # Obtain the associated image of the BBDD of each query image
    if args.m == 'd':
        with open(args.q / "gt_corresps.pkl", 'rb') as f:
            data = pickle.load(f)

    # Obtain the number of images on the database
    n = len(glob.glob1(args.p, "*.jpg"))

    # Obtain the number of query images
    t = len(glob.glob1(args.q, "*.jpg"))

    # List of lists
    exp_euclidean = []
    exp_intersection = []
    exp_l1 = []
    exp_chi2 = []
    exp_hellinger = []

    for j in tqdm(range(t)):

        # Read the query image
        img_file = args.q.as_posix() + '/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
        img = cv2.imread(img_file)

        # Obtain the hist of the query image depending on the color space
        hist = computeHistImage(img, color_space=args.c)

        eucl_distances = np.array([])
        instersection_distances = np.array([])
        l1_distances = np.array([])
        chi2_distances = np.array([])
        hellinger_distances = np.array([])

        for i in range(n):

            # Read the database image
            db_file = args.p.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
            db_img = cv2.imread(db_file)

            # Obtain the hist of the database image depending on the color space
            db_hist = computeHistImage(db_img, color_space=args.c)

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

            elif args.d == "chi2":
                chi2_distances = np.append(chi2_distances, computeSimilarity(hist, db_hist, similarity_measure='chi2'))

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
        print('mAP@k (K = {}) of the desired distances'.format(int(args.k)))

        if args.d == "all":
            print("Euclidean Distance: {0:.4f}".format(mapk(data, exp_euclidean, args.k)))
            print("Histogram Intersection: {0:.4f}".format(mapk(data, exp_intersection, args.k)))
            print("L1 Distance: {0:.4f}".format(mapk(data, exp_l1, args.k)))
            print("Chi-Squared Distance: {0:.4f}".format(mapk(data, exp_chi2, args.k)))
            print("Hellinger Distance: {0:.4f}".format(mapk(data, exp_hellinger, args.k)))

        elif args.d == "euclidean":
            print("Euclidean Distance: {0:.4f}".format(mapk(data, exp_euclidean, args.k)))

        elif args.d == "intersec":
            print("Histogram Intersection: {0:.4f}".format(mapk(data, exp_intersection, args.k)))

        elif args.d == "chi2":
            print("Chi-Squared Distance: {0:.4f}".format(mapk(data, exp_chi2, args.k)))

        elif args.d == "l1":
            print("L1 Distance: {0:.4f}".format(mapk(data, exp_l1, args.k)))

        elif args.d == "hellinger":
            print("Hellinger Distance: {0:.4f}".format(mapk(data, exp_hellinger, args.k)))

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