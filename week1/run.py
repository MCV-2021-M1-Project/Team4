import argparse
import pickle
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from mapk import mapk

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-k', '--k',  type=int, default=10, help='The K most similar images to the query one')
    #parser.add_argument('-c', '--color_space', type="str", help='Color Space in which the histograms will be computed')
    #parser.add_argument('-d', '--distance', type="str", help='Distance to compare the histograms')
    return parser.parse_args()


def euclidean_distance(u,v):
    return np.linalg.norm(u - v)


def main():
    args = parse_args()
    # Obtain the associated image of the BBDD of each query image
    # Go in the data directory -> qsd1_w1
    db_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/data"
    with open(db_path + '/qsd1_w1/gt_corresps.pkl', 'rb') as f:
        data = pickle.load(f)

    # Obtain the number of query images
    t = len(glob.glob1(db_path + '/qsd1_w1', "*.jpg"))

    # Obtain the number of images on the database
    n = len(glob.glob1(db_path + '/BBDD', "*.jpg"))

    exp_distances = []

    for j in tqdm(range(t)):

        # Read the query image
        img_file = db_path + '/qsd1_w1/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
        img = cv2.imread(img_file)

        # HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
        hist = hist / hsv_img[:, :, 0].size

        distances = np.array([])

        for i in range(n):

            # Read the database image
            db_file = db_path + '/BBDD/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
            db_img = cv2.imread(db_file)

            # HSV color space
            db_hsv_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2HSV)
            db_hist = cv2.calcHist([db_hsv_img], [0], None, [256], [0, 256])
            db_hist = db_hist / db_hsv_img[:, :, 0].size

            # Euclidean distances
            dist = euclidean_distance(hist, db_hist)
            distances = np.append(distances, dist)

        exp_distances.append(distances.argsort(axis=0)[:args.k].tolist())

    print('Euclidean Distance MAPK (K = {})'.format(int(args.k)))
    print(mapk(data,exp_distances,args.k))







if __name__ == "__main__":
    main()