import cv2

BBDD_PATH = '/home/david/Desktop/M1/data/BBDD/'
QUERY_PATH = '/home/david/Desktop/M1/data/BBDD/'

bfmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

