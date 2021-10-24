import glob
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import os
from multiresolution import multiresolution
from background_substraction import substractBackground

import matplotlib.pyplot as plt

# -- CHECK ARGUMENTS --

def checkArguments(args):
    if args.m != 'd' and args.m != 't':
        raise TypeError("The modes of the code are: development(d) and test(t)")
    if args.c not in ["GRAY", "RGB", "HSV", "YCrCb", "CIELab"]:
        raise TypeError("Wrong color space")
    if args.d not in ["all", "euclidean", "intersec", "l1", "chi2", "hellinger", "chi2alt"]:
        raise TypeError("Wrong distance")
    if args.m == 't' and args.d == 'all':
        raise Exception("The test mode cannot be done with all the distances at the same time given that only one "
                        "distance can be stored in the same file")

# -- PREPROCESSING --

def equalizeImage(img):
    # Equalizing Saturation and Lightness via HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    eqV = cv2.equalizeHist(v)
    img = cv2.merge((h, s, eqV))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img



def get_histograms_from_set(path_set, args):
    data_path = str(path_set).split('data')[0]+'data'
    set_name = str(path_set).split('data')[1]
    hists_path = data_path + '/histograms'
    query_data_path = data_path + '/' + set_name
    histogram_list_pkl = hists_path + '/' + set_name + '_' + args.c + '_' + args.rt+ '_' + str(args.r) +'.pkl'

    # Create dir if not exists
    if(not os.path.exists(hists_path)):
        os.mkdir(hists_path)

        # Retrieve list or create it
        if (os.path.exists(histogram_list_pkl)):
            print('Loading histogram list from set: ' + set_name)
            with open(histogram_list_pkl, 'rb') as f:
                return pickle.load(f)
    else:
        n = len(glob.glob1(args.q, "*.jpg"))
        # if args.b == "y" and not 'BBDD' in set_name:
        mask, coords = substractBackground(numImages=n, args=args)

        print('Computing the histograms of all the images of the set: ' + set_name)
        files = [f for f in os.listdir(query_data_path) if (f.endswith('.jpg'))]
        hist_list = []
        j = 0
        for filename in tqdm(files):
            filename = os.fsdecode(filename)
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(query_data_path, filename))

                try:
                    # Interseccion
                    m = mask[j]
                except:
                    m = None
                #img = equalizeImage(img)
                # Append all the query images histograms
                hist_list.append(multiresolution(img, color_space=args.c, level=args.r, type=args.rt, mask=m))

            else:
                continue
            j+=1

        # Store list in pkl
        with open(histogram_list_pkl, 'wb') as f:
            pickle.dump(hist_list, f)

    return hist_list