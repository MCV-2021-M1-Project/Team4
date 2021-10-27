import numpy as np
import pytesseract
import cv2

img = cv2.imread('00008.jpg')
extractedInformation = pytesseract.image_to_string(img)

print(extractedInformation)