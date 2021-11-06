import cv2


def feature_descriptor(img, type='sift', mask=None):
    if type == 'sift':
        sift = cv2.SIFT_create(800)
        keypoints, descriptors = sift.detectAndCompute(img, mask)

    # Este va peor pero es mas rapido
    elif type == 'orb':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img, mask)

    return descriptors


def check_painting_db(distances_i, image_distances, th=0.03):
    # This function modifies the vector image distances if detects that a painting is not in the database
    i = 0
    for bbdd_number in image_distances:
        if distances_i[bbdd_number] > th:
            image_distances.insert(i, -1)         # -1 means that there is no painting
            del image_distances[-1]                 # Drop last number
            return image_distances
        i += 1
    return image_distances





