import cv2
import matplotlib.pyplot as plt
from skimage import feature
import numpy as np
from similarities import hellingerDistance
from LBP_descriptor import crop_image, LBP_blocks
from colorDescriptors import colorDescriptors
from dct_descriptor import dct_descriptor


def combinedDescriptors(img, args, mask=None):
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(mask)
    # plt.show()

    # Compute color space descriptor
    if args.color is not None:
        color_descriptor = colorDescriptors(img, block=args.l, color_space=args.color, mask=mask)
    else:
        color_descriptor = np.array([])

    # Compute texture descriptors
    if args.texture is not None:
        if args.texture == "LBP":
            texture_descriptor = LBP_blocks(img, p=args.pe, r=args.r, block=args.l, mask=mask)
        elif args.texture == "DCT":
            texture_descriptor = dct_descriptor(img, mask)
    else:
        texture_descriptor = np.array([])

    # Concatenate texture and color (not that CDT is not a histogram!)
    if not args.texture == 'DCT':
        color_texture_descriptor = np.concatenate((color_descriptor, texture_descriptor))
    else:
        color_texture_descriptor = [color_descriptor, texture_descriptor]

    return color_texture_descriptor








