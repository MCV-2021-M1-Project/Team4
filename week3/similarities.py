import cv2

# FILE WITH ALL THE FUNCTIONS WHICH DEAL WITH COMPUTING THE SIMILARITIES BETWEEN DESCRIPTORS

def hellingerDistance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
