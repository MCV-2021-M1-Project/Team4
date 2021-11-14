import cv2
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def order_points_old(pts):
    # Function to order the BBox points in clockwise order
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def rotate_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < 35 and angle > -35:
            angles.append(angle)

    if len(angles) == 0:
        median_angle = 0
    else:
        median_angle = np.median(angles)

    img_rotated = ndimage.rotate(img, median_angle, mode='nearest')

    return img_rotated, median_angle

def find_frame_points(mask, angle):
    mask_rotated = ndimage.rotate(mask, -angle, mode='nearest')

    dst = cv2.cornerHarris(mask_rotated, 5, 3, 0.06)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(dst, np.float32(centroids[1:]), (5, 5), (-1, -1), criteria)
    # Order the corners in clockwise order
    corners = order_points_old(corners)

    # Convert corners array to list
    corners = corners.tolist()

    # Convert corners array to list
    corners = [[int(float(point)) for point in corner] for corner in corners]

    #for idx, corner in enumerate(corners):
    #   cv2.putText(dst, str(idx + 1), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255),5)

    return corners

