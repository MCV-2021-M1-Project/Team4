import cv2
import numpy as np
import matplotlib.pyplot as plt
from noise_detection_and_removal import remove_noise
from text_box import bounding_box

# Prueba query set w3, una imagen solo

img_query = cv2.imread('/Users/Cesc47/Documents/CesC_47/MCV/M1/data/qsd1_w3/00000.jpg')
img_bbdd = cv2.imread('/Users/Cesc47/Documents/CesC_47/MCV/M1/data/BBDD/bbdd_00075.jpg')

img_query = remove_noise(img_query)
"""
[left, top, right, bottom] = bounding_box(img_query)
mask1 = np.ones(img_query.shape[:2], dtype=np.uint8)
mask1[top:bottom, left:right] = 0
"""

img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
img_bbdd = cv2.cvtColor(img_bbdd, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_query, descriptors_query = sift.detectAndCompute(img_query, None)
keypoints_bbdd, descriptors_bbdd = sift.detectAndCompute(img_bbdd, None)

# Method 1 to find the matches: norm_l1 and comparing everyone
# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# Compute matches, sort and discard some of them (with high distances)
matches = bf.match(descriptors_query, descriptors_bbdd)
matches = sorted(matches, key = lambda x:x.distance)

matches_ok = []
threshold = 3000 # Threshold of L1 distance
for match in matches:
    if match.distance < threshold:
        matches_ok.append(match)
# Hacer ponderacion distancia con el match + numero de distancias para caso base de datos
img3 = cv2.drawMatches(img_query, keypoints_query, img_bbdd, keypoints_bbdd, matches_ok, img_bbdd, flags=2)

plt.imshow(img3)
plt.show()