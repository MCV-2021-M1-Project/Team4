import cv2
import numpy as np

# Mirar ltp, es mejor que lbp!
def texture_descriptors(img, mask=None, type='dct', plot_results=False):
    if type == 'dct':
        img = img[:, :, 2]
        mblock_size = 8
        num_mblocks_x = int(img.shape[0] / mblock_size)
        num_mblocks_y = int(img.shape[1] / mblock_size)
        imf = np.float32(img) / 255.0  # float conversion/scale

        dct = np.zeros([num_mblocks_x*mblock_size, num_mblocks_y*mblock_size])
        for i in range(num_mblocks_x):
            for j in range(num_mblocks_y):
                dst = cv2.dct(imf[mblock_size*i:mblock_size*i+mblock_size, mblock_size*j:mblock_size*j+mblock_size])
                dct[mblock_size*i:mblock_size*i+mblock_size, mblock_size*j:mblock_size*j+mblock_size] = dst
        dct = np.uint8(dct*255)   # convert back

        return dct.flatten()



'''
t = 5
for j in range(t):
    img_file = '/Users/Cesc47/Documents/CesC_47/MCV/M1/data/qsd1_w3' + '/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coefs = texture_descriptors(img, plot_results=True)
'''