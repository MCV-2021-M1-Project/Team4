import cv2
import numpy as np
import matplotlib.pyplot as plt
from noise_detection_and_removal import remove_noise
from text_box import bounding_box
from background_substraction import substractBackground

# ----- WHEN THE BBDD IMAGE AND THE QUERY IMAGE ARE THE SAME -----

img1 = cv2.imread('/home/david/Desktop/M1/data/qsd1_w4/00015.jpg')
img1 = remove_noise(img1)
mask = substractBackground(img1)[2]

[left, top, right, bottom] = bounding_box(img1)
mask[top:bottom, left:right] = 0

img2 = cv2.imread('/home/david/Desktop/M1/data/BBDD/bbdd_00182.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.ORB_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, mask)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

print(f"Number of keypoints bbdd: {len(keypoints_2)}")
print(f"Number of keypoints qsd1: {len(keypoints_1)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        print(m.distance)


print(f"matches: {len(good)}, {len(good)/len(keypoints_1)}%")

img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,good, None, flags=2)

plt.imshow(img3)
plt.show()

# ----- WHEN THE BBDD IMAGE AND THE QUERY IMAGE ARE DIFFERENT -----

img1 = cv2.imread('/home/david/Desktop/M1/data/qsd1_w4/00006.jpg')
img1 = remove_noise(img1)
mask = substractBackground(img1)[0]

[left, top, right, bottom] = bounding_box(img1)
mask[top:bottom, left:right] = 0

img2 = cv2.imread('/home/david/Desktop/M1/data/BBDD/bbdd_00250.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.ORB_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, mask)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

print(f"Number of keypoints bbdd: {len(keypoints_2)}")
print(f"Number of keypoints qsd1: {len(keypoints_1)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        print(m.distance)


print(f"matches: {len(good)}, {len(good)/len(keypoints_1)}%")

img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,good, None, flags=2)

plt.imshow(img3)
plt.show()