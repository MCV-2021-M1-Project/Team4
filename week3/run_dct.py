import argparse
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
from dct_descriptor import dct_descriptor
from histograms import computeSimilarity
from mapk import mapk



def parse_args():
    """
    Function to get the input arguments
    Returns
    parse_args()
    """
    parser = argparse.ArgumentParser(description='DCT computation')
    parser.add_argument('-k', type=int, default=10,
                        help='Number of images to retrieve')
    parser.add_argument('-p', type=Path, required=True,
                        help='Path to the database directory')
    parser.add_argument('-q', type=Path, required=True,
                        help='Path to the query set directory')
    return parser.parse_args()


def main():
    # Obtain the arguments
    args = parse_args()

    with open(args.q / "gt_corresps.pkl", 'rb') as f:
        data = pickle.load(f)

    # Obtain the number of images on the database
    n_query_images = len(glob.glob1(args.q, "*.jpg"))

    # Obtain the number of query images
    n_bbdd_images = len(glob.glob1(args.p, "*.jpg"))

    # FIRST: compute the DCT of the database
    bbdd_dcts = []
    for bbdd in tqdm(range(n_bbdd_images)):
        bbdd_file = ('0000' if bbdd < 10 else '000' if bbdd < 100 else '00') + str(bbdd)
        bbdd_img = cv2.imread(str(args.p) + '/bbdd_' + bbdd_file + '.jpg')
        
        bbdd_dct = dct_descriptor(bbdd_img)
        bbdd_dcts.append(bbdd_dct)

    # SECOND: Iterate on the query images to:
    #   - Compute the DCT of the query images
    #   - Iterate on the BBDD DCT data to compute the Euclidean distance with all the BBDD images
    #   - Sort the best images with best similarities
    #   - Compute the mapk of the results
    
    all_similarities = []
    for query in tqdm(range(n_query_images)):
        query_file = ('0000' if query < 10 else '000' if query < 100 else '00') + str(query)
        query_img = cv2.imread(str(args.q) + '/' + query_file + '.jpg')
        
        query_dct = dct_descriptor(query_img)
        
        query_similarities = []
        for bbdd in range(n_bbdd_images):
            similarity = computeSimilarity(query_dct,bbdd_dcts[bbdd],'euclidean')
            query_similarities.append(similarity)

        all_similarities.append(np.argsort(query_similarities)[:int(args.k)].tolist())
        

    # Compute and print the mAP
    result = mapk(data,all_similarities,int(args.k))
    print('MAP@' + str(args.k) + ': ' + str(result))


if __name__ == "__main__":
    main()
