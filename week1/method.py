import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import os
from mapk import mapk
from utils import computeHistImage, computeSimilarity

def main(color_space, similarity_measure):
    # Data path
    db_path = "../../data"
    k = 10
    # Open Ground Truth picke file
    with open(db_path + '/qsd1_w1/gt_corresps.pkl', 'rb') as f:
        data = pickle.load(f)

    # Loop Task 1 query dir
    query_set_path = db_path + '/qsd1_w1/'
    db_set_path = db_path + '/BBDD/'
    exp_distances = []

    count = 0
    for file_query in os.listdir(query_set_path):
        if count > 2:
            break
        count +=1
        if file_query.endswith(".jpg"):
            # Read query
            img = cv2.imread(query_set_path + file_query)
            # Compute normalized histogram
            img_hist = computeHistImage(img, color_space)
            distances = np.array([])
            for file_db in os.listdir(db_set_path):
                if file_db.endswith(".jpg"):
                    db_img = cv2.imread(db_set_path + file_db) # Read image from db
                    db_img_hist = computeHistImage(db_img, color_space) # Compute norm hist

                    # Compute similarity
                    dist = computeSimilarity(img_hist, db_img_hist, similarity_measure)
                    print(file_db, dist)
                    distances = np.append(distances, dist)

            k_images_dist = distances.argsort(axis=0)[:k].tolist()
            print("dist", k_images_dist)
            exp_distances.append(k_images_dist)

    print(similarity_measure,  " Distance MAPK: ")
    print(mapk(data, exp_distances, k))
   

if __name__ == "__main__":
    color_space = 'RGB'
    similarity_measure = 'euclidean'
    main(color_space, similarity_measure)


