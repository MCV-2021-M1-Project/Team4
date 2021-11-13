import numpy as np
import pickle
import cv2
import math
import imutils
from scipy import ndimage
import matplotlib.pyplot as plt
from noise_detection_and_removal import remove_noise
from background_substraction import substractBackground
from evaluation import angular_error
from rotation import find_frame_points

with open('/home/david/Desktop/M1/data/qsd1_w5/frames.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"IMAGE NUM\t| GROUND TRUTH ANGLE\t| PREDICTED ANGLE\t    | ANGULAR ERROR ")
print(f"-------------------------------------------------------")
for i in range(9, 30):
    img_file = '/home/david/Desktop/M1/data/qsd1_w5/' + '/00' + ('00' if i < 10 else '0') + str(i) + '.jpg'

    img = cv2.imread(img_file)
    img = remove_noise(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []
    # Iterar en todas las linias encontradas
    for [[x1, y1, x2, y2]] in lines:
        # Dibujar linias encontradas en la imagen
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Calcular angulo de las linias
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Quedarse solo con las lineas que sean horizontales
        if angle < 45 and angle > -45:
            angles.append(angle)
    plt.imshow(img)
    plt.show()
    # Angulo medio
    median_angle = np.median(angles)

    # Rotar imagen con nearest neighbor en los bordes para facilitar background substraction
    img_rotated = ndimage.rotate(img, median_angle, mode='nearest')
    plt.imshow(img_rotated)
    plt.show()

    # Cambio de conversion de [-90, 90] a [0,180]
    """
    if median_angle > 0:
        median_angle = 180 - median_angle
    elif median_angle < 0:
        median_angle = -median_angle
    """

    # Calcular background con plot==TRUE para ver como funciona cada mascara(OPENING, GRADIENT Y CANNY)
    masks = substractBackground(img_rotated, plot=True)
    #for mask in masks:
    #    rotate_mask(mask, median_angle)

    corners = find_frame_points(masks[0], median_angle)
    print(f"{data[i][0][1]} - {corners}")

    changed_angle = 0
    if median_angle > 0:
        changed_angle = 180 - median_angle
    elif median_angle < 0:
        changed_angle = -median_angle

    print(f"\t{i:02d}      |\t    {data[i][0][0]:.04f}\t        |\t   {changed_angle:.04f}\t        |\t   {angular_error(data[i][0][0], median_angle):.04f}")
