import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle


from mapk import mapk

n = 287 #Number of museum images
t = 30 #Number of queries
k = 10 #Number of most similar images

with open('./qsd1_w1/gt_corresps.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

exp_distances = []
exp_intersections = []
exp_chi = []
exp_kernels = []

#Reading the image into numpy array
for j in range(t):
    print(j)

    img_file =  db_file = './qsd1_w1/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
    img = cv2.imread(img_file)

    #HSV color space
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # hist = np.concatenate((cv2.calcHist([ycrcb_img], [2], None, [256], [0,256]),
    #                       cv2.calcHist([ycrcb_img], [1], None, [256], [0,256])),
    #                      axis=0)
    hist = cv2.calcHist([ycrcb_img], [2, 1], None, [256, 256], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    distances = np.array([])
    intersections = np.array([])
    chisquared = np.array([])
    kernels = np.array([])

    for i in range(n):
        #Museum DB files
        db_file = './BBDD/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.jpg'
        
        db_img = cv2.imread(db_file)

        #XYZ color space
        db_ycrcb_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2YCrCb)
        db_hist = cv2.calcHist([db_ycrcb_img], [2, 1], None, [256, 256], [0, 256, 0, 256])
        db_hist = cv2.normalize(db_hist, db_hist).flatten()
        """
        db_ycrcb_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2YCrCb)
        db_hist = np.concatenate((cv2.calcHist([db_ycrcb_img], [2], None, [256], [0, 256]),
                                  cv2.calcHist([db_ycrcb_img], [1], None, [256], [0, 256])),
                                 axis=0)

        db_hist = db_hist/(db_ycrcb_img[:,:,2].size + db_ycrcb_img[:,:,1].size)
        """
        #Euclidean distances
        dist = np.linalg.norm(hist - db_hist)
        distances = np.append(distances,dist)

        #Histogram intersection
        inter = np.sum(np.minimum(hist,db_hist))
        intersections = np.append(intersections,inter)

        #Chi-Squared distance
        chi = cv2.compareHist(hist, db_hist, cv2.HISTCMP_CHISQR)
        chisquared = np.append(chisquared, chi)

        #Hellinger kernel
        kernel = np.sum(np.sqrt(hist,db_hist))
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

    print('\nMinimum Chi-Squered distance images')
    k_images_chi = np.flip(kernels.argsort(axis=0)[-k:]).tolist()
    print(k_images_chi)
    print("\n")

    exp_chi.append(k_images_chi)

    print('\nMaximum Hellinger kernel images')
    k_images_kernel = chisquared.argsort(axis=0)[:k].tolist()
    print(k_images_kernel)
    print("\n")

    exp_kernels.append(k_images_kernel)

print("Euclidean Distance MAPK: ")
print(mapk(data,exp_distances,k))

print("Histogram Intersection MAPK: ")
print(mapk(data,exp_intersections, k))

print("ChiSquared Distance Distance MAPK: ")
print(mapk(data,exp_chi, k))

print("Hellinger kernel MAPK: ")
print(mapk(data,exp_kernels, k))

