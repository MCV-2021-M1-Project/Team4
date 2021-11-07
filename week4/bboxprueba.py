import glob
import cv2
from noise_detection_and_removal import remove_noise
from background_substraction import substractBackground
from text_box import bounding_box
import matplotlib.pyplot as plt

QUERY_PATH = '/home/david/Desktop/M1/data/qsd1_w4/'

n_bbdd_images = len(glob.glob1(QUERY_PATH, "*.jpg"))

for i in range(n_bbdd_images):
    img_file = QUERY_PATH + ('0000' if i < 10 else '000') + str(i) + '.jpg'
    img = cv2.imread(img_file)
    img = remove_noise(img)

    masks = substractBackground(img)

    for idx, mask in enumerate(masks):
        [left, top, right, bottom] = bounding_box(img)
        mask[top:bottom, left:right] = 0
        f, ax = plt.subplots(1,2)
        plt.title(f"{i} - {idx}")
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[1].imshow(mask)
        plt.show()






