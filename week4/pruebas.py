import cv2
import numpy as np
import matplotlib.pyplot as plt
from noise_detection_and_removal import remove_noise
from text_box import bounding_box

# ----- WHEN THE BBDD IMAGE AND THE QUERY IMAGE ARE THE SAME -----

img1 = cv2.imread('/home/david/Desktop/M1/data/qsd1_w3/00000.jpg')
img1 = remove_noise(img1)
[left, top, right, bottom] = bounding_box(img1)
mask1 = np.ones(img1.shape[:2],dtype=np.uint8)
mask1[top:bottom, left:right] = 0

img2 = cv2.imread('/home/david/Desktop/M1/data/BBDD/bbdd_00092.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, mask1)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

print(f"Number of keypoints bbdd: {len(keypoints_2)}")
print(f"Number of keypoints qsd1: {len(keypoints_1)}")

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

print(f"matches: {len(matches)}")

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

plt.imshow(img3)
plt.show()

# ----- WHEN THE BBDD IMAGE AND THE QUERY IMAGE ARE DIFFERENT -----

img1 = cv2.imread('/home/david/Desktop/M1/data/qsd1_w3/00000.jpg')
img1 = remove_noise(img1)
[left, top, right, bottom] = bounding_box(img1)
mask1 = np.ones(img1.shape[:2],dtype=np.uint8)
mask1[top:bottom, left:right] = 0

img2 = cv2.imread('/home/david/Desktop/M1/data/BBDD/bbdd_00094.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, mask1)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

print(f"Number of keypoints bbdd: {len(keypoints_2)}")
print(f"Number of keypoints qsd1: {len(keypoints_1)}")

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

print(f"matches: {len(matches)}")

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

plt.imshow(img3)
plt.show()