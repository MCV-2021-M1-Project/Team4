import cv2
import math
import numpy as np
from scipy import ndimage


def rotate_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < 45 and angle > -45:
            angles.append(angle)

    median_angle = np.median(angles)

    img_rotated = ndimage.rotate(img, median_angle, mode='nearest')

    return img_rotated, median_angle