import cv2
import glob
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from noise_detection_and_removal import remove_noise
from text_box import bounding_box
from evaluation import mapk

BBDD_PATH = '/home/david/Desktop/M1/data/BBDD/'
n_bbdd_images = len(glob.glob1(BBDD_PATH, "*.jpg"))

QUERY_PATH = '/home/david/Desktop/M1/data/qsd1_w3/'
n_query_images = len(glob.glob1(QUERY_PATH, "*.jpg"))

with open(QUERY_PATH + "gt_corresps.pkl", 'rb') as f:
    data = pickle.load(f)

# Create SIFT
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

bbdd_kps = []
bbdd_descs = []

"""
for i in tqdm(range(n_bbdd_images)):
    img_file = BBDD_PATH + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(gray_img, None)
    bbdd_descs.append(desc)

with open('SIFTdescriptors.pkl', 'wb') as f:
    pickle.dump(bbdd_descs, f)
"""
with open("/home/david/Desktop/M1/Team4/week4/SIFTdescriptors.pkl", 'rb') as f:
    descriptors = pickle.load(f)

arg = []
for i in tqdm(range(n_query_images)):
    img_file = QUERY_PATH + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'
    img = cv2.imread(img_file)
    img = remove_noise(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    [left, top, right, bottom] = bounding_box(img)
    mask = np.ones(img.shape[:2], dtype=np.uint8)
    mask[top:bottom, left:right] = 0

    query_kp, query_desc = sift.detectAndCompute(gray, mask)

    number_of_matches = []
    for bbdd_desc in descriptors:
        if bbdd_desc is not None:
            matches = bf.knnMatch(query_desc, bbdd_desc, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            number_of_matches.append(len(good))
        else:
            number_of_matches.append(0)

    arg.append(np.argsort(np.array(number_of_matches))[::-1].tolist())

print(f'mAP@k (k=10)')
print(mapk(data, arg, 10))











"""
all_matches = []
for bbdd_desc, bbdd_kp in zip(bbdd_descs, bbdd_kps):
    thresholded_matches = []
    matches = bf.match(bbdd_desc, query_desc)
    matches = sorted(matches, key = lambda x:x.distance)
    for match in matches:
        if match.distance < THRESHOLD:
            thresholded_matches.append(match)

    all_matches.append(thresholded_matches)
"""