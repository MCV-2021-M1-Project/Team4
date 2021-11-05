import cv2
import glob
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from noise_detection_and_removal import remove_noise
from text_box import bounding_box

BBDD_PATH = '/home/david/Desktop/M1/data/BBDD/'
n_bbdd_images = len(glob.glob1(BBDD_PATH, "*.jpg"))

# Create SIFT
sift = cv2.SIFT_create()

bbdd_kps = []
bbdd_descs = []

for i in tqdm(range(n_bbdd_images)):
    img_file = BBDD_PATH + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(gray_img, None)
    bbdd_kps.append(kp)
    bbdd_descs.append(desc)

with open('KeyPoints.pkl', 'wb') as f:
    pickle.dump(bbdd_kps, f)

with open('Descriptors.pkl', 'wb') as f:
    pickle.dump(bbdd_descs, f)
"""  
img = cv2.imread('/home/david/Desktop/M1/data/qsd1_w3/00000.jpg')
img = remove_noise(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
[left, top, right, bottom] = bounding_box(img)
mask1 = np.ones(img.shape[:2], dtype=np.uint8)
mask1[top:bottom, left:right] = 0

query_kp, query_desc = sift.detectAndCompute(gray_img, mask1)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = []
percentage_matches = []
for bbdd_desc, bbdd_kp in zip(bbdd_descs, bbdd_kps):
    m = bf.match(bbdd_desc, query_desc)
    m = sorted(m, key = lambda x:x.distance)
    matches.append(m)
    if len(bbdd_kp) == 0:
        percentage_matches.append(0)
    else:
        percentage_matches.append(len(matches)/(len(bbdd_kp)*len(query_kp)))

arg_distances = np.argsort(percentage_matches).tolist()

print(arg_distances)

"""