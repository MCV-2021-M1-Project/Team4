import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle


from mapk import mapk
from utils import euclidean_distance, histogram_intersection, hellinger_kernel

n = 287 #Number of museum images
t = 30 #Number of queries
k = 10 #Number of most similar images

db_path = "../../data"
with open(db_path + '/qsd1_w1/gt_corresps.pkl', 'rb') as f:
    data = pickle.load(f)

exp_distances = []
exp_intersections = []
exp_kernels = []

#Reading the image into numpy array
for j in range(t):
    print(j)

    img_file = db_path + '/qsd1_w1/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
    img = cv2.imread(img_file)
    #HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0], None, [256], [0,256])
    hist = hist/hsv_img[:,:,0].size

    distances = np.array([])
    intersections = np.array([])
    kernels = np.array([])

    for i in range(n):
        #Museum DB files
        db_file = db_path + '/BBDD/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
        
        db_img = cv2.imread(db_file)

        #XYZ color space
        db_hsv_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2HSV)
        db_hist = cv2.calcHist([db_hsv_img], [0], None, [256], [0,256])
        db_hist = db_hist/db_hsv_img[:,:,0].size

        #Euclidean distances
        dist = euclidean_distance(hist,db_hist)
        distances = np.append(distances,dist)

        #Histogram intersection
        inter = histogram_intersection(hist,db_hist)
        intersections = np.append(intersections,inter)

        #Hellinger kernel
        kernel = hellinger_kernel(hist,db_hist)
        kernels = np.append(kernels,kernel)


    print('\nMinimum Euclidean distance images')
    k_images_eucl = distances.argsort(axis=0)[:k].tolist()
    print(k_images_eucl)

    exp_distances.append(k_images_eucl)

    print('\nMaximum histogram intersection images')
    k_images_inter = np.flip(intersections.argsort(axis=0)[-k:]).tolist()
    print(k_images_inter)
    print("\n")

    exp_intersections.append(k_images_inter)

    print('\nMaximum Hellinger kernel images')
    k_images_kernel = np.flip(kernels.argsort(axis=0)[-k:]).tolist()
    print(k_images_kernel)
    print("\n")

    exp_kernels.append(k_images_kernel)


print("Euclidean Distance MAPK: ")
print(mapk(data,exp_distances,k))

print("Histogram Intersection MAPK: ")
print(mapk(data,exp_intersections, k))

print("Hellinger kernel MAPK: ")
print(mapk(data,exp_kernels , k))
