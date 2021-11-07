import numpy as np
import pytesseract

# Only windows
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

import cv2
import matplotlib.pyplot as plt
import glob
import re
import textdistance as td
import ast

from pathlib import Path

from text_box import bounding_box


def read_text(img, bbox):
    # Coordinates of the BBox
    [left, top, right, bottom] = bbox

    text_img = img[top:bottom, left:right]  # Crop image
    extractedText = pytesseract.image_to_string(text_img)

    # If we don't find text in the image, we try with a bigger bbox
    if extractedText[:-1] == '':
        top = top - 5 if top - 5 > 0 else 0
        bottom = bottom + 5 if bottom + 5 < img.shape[0] else 0
        text_img = img[top:bottom, left:right]
        extractedText = pytesseract.image_to_string(text_img)
    # If we don't find text in the image, we try with a smaller bbox
    if extractedText[:-1] == '':
        if bottom - top > 20:
            top = top + 10
            bottom = bottom - 10
            text_img = img[top:bottom, left:right]
            extractedText = pytesseract.image_to_string(text_img)

    # Delete extrange chars that never appear in the pictures. Ex.: \n, \r, \x0C, =, %, @, &.
    extractedText = extractedText.replace("\n", "").replace("\r", "").replace("\x0C", "").replace("=", "").replace("%",
                                                                                                                   "").replace(
        "@", "").replace("&", "")

    # Delete  white spaces in the extrem
    while len(extractedText) and extractedText[0] == ' ':
        extractedText = extractedText[1:]

    while len(extractedText) and extractedText[-1] == ' ':
        extractedText = extractedText[:-1]

    return extractedText


def extractTextGroundTruth(path:Path):
    """
    Find the text (painter and tilte) of all the images of the database
    :param path: path to the database directory
    :return:
        text_corresp:   Dictionary in which all the titles and painters are stored as keys and a list of all the images
                        that contain that key
        text_data:      List with all the keys (titles and painters)
    """
    # Find the number of images in the BBDD directory
    n_bbdd_images = len(glob.glob1(path, "*.txt"))

    # Dictionary in which the painter and painting names are stored as keys and a list with the images which have this
    # title or painter
    text_corresp = {}

    # Iterate in all the BBDD images
    for i in range(n_bbdd_images):
        txt_file = path.as_posix() + '/bbdd_00' + ('00' if i < 10 else ('0' if i < 100 else '')) + str(i) + '.txt'

        # Open the text file
        with open(txt_file) as f:
            first_line = f.readline()

        # The title and painter are between simple quotes, so search the strings between single quotes
        titles = re.findall(r"'([^']+)'", first_line)

        # Remove the quotes from the titles
        for title in titles:
            tit_without_quotes = title.replace("'", "")

            # If in the dictionary there is not a key with this title name, create the new key and the new list in which
            # the images with the same key will be stored
            if text_corresp.get(tit_without_quotes) is None:
                text_corresp[tit_without_quotes] = [i]

            # If the key already exists, append the image to the list corresponding to that title
            else:
                list_paintings = text_corresp[title]
                list_paintings.append(i)
                text_corresp[title] = list_paintings

    # Create a list of all the keys (titles and painters) that exist
    text_data = list(text_corresp.keys())

    return text_corresp, text_data


def compareArguments(arg_distances, text, text_corresp, text_data):
    """
    This function modifies the list of already sorted images by a descriptor taking into account the read text
    :param arg_distances: List of sorted database images number by a descriptor
    :param text: read text from the BBox part of the image
    :param text_corresp: Dictionary in which all the titles and painters are stored as keys and a list of all the images
                        that contain that key
    :param text_data: List with all the keys (titles and painters)
    :return: Modified list of sorted database images number by a descriptor (arg_distances)
    """

    # Compare with the Levenshtein distance the read text with all the keys of text_correspondences
    text_distances = []
    for t in text_data:
        text_distances.append(td.levenshtein.distance(text, t))

    # Sort the distances to obtain the lowest distances in the firsts positions
    arg_text = np.argsort(text_distances).tolist()

    # Take the text with lowest distance to string
    predicted_text = text_data[arg_text[0]]

    # print(arg_distances)
    # print(text + '->' + predicted_text)

    # If the distance of the best retrieval divided by the length of the title is less than a threshold, then:
    # Search the first occurring item on the arg_distances (sorted painting numbers after a descriptor has already done
    # the retrieval), remove the item and put it in the first position.
    if (np.min(text_distances) / len(predicted_text)) < 0.55:
        pred = next(pred for pred in arg_distances if pred in text_corresp[predicted_text])
        arg_distances.remove(pred)
        arg_distances.insert(0, pred)

    # print(arg_distances)
    # print()
    # Return the modified arg_distances
    return arg_distances, predicted_text