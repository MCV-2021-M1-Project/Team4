import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle


def dct_descriptor(img, mask = None):

    dst_width = 500
    dst_height = 500
    pts_dst = np.array([[0,0],[0,dst_width],[dst_height,0],[dst_height,dst_width]])
    size = (dst_width,dst_height)

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    width = img_gray.shape[0]
    height = img_gray.shape[1]

    pts_src = np.array([[0,0],[0,width],[height,0],[height,width]])
    pts = [pts_src,pts_dst]
    
    im_out = homographic_transform(img_gray,size,pts)
    
    dct_transform = dct(np.float32(im_out)/255.0)

    return dct_transform

def dct(img):
    
    mblock_size = 8
    num_mblocks_x = int(np.ceil(img.shape[0] / mblock_size))
    num_mblocks_y = int(np.ceil(img.shape[1] / mblock_size))
    mirror_x = mblock_size - np.mod(img.shape[0], mblock_size) # Number of rows to apply mirroring
    mirror_y = mblock_size - np.mod(img.shape[1], mblock_size) # Number of columns to apply mirroring
    
    # Mirroring
    img_mirrored = np.zeros((num_mblocks_x*mblock_size,num_mblocks_y*mblock_size))
    img_mirrored[:img.shape[0],:img.shape[1]] = img
    img_mirrored[-mirror_x:,:] = img_mirrored[-2*mirror_x:-mirror_x,:][::-1]
    img_mirrored[:,-mirror_y:] = np.flip(img_mirrored[:,-2*mirror_y:-mirror_y])[::-1]
    
    #Apply DCT per blocks
    dct = np.zeros([num_mblocks_x*mblock_size, num_mblocks_y*mblock_size])
    for i in range(num_mblocks_x):
        for j in range(num_mblocks_y):
            dct_block = cv2.dct(img_mirrored[i*mblock_size:(i+1)*mblock_size,j*mblock_size:(j+1)*mblock_size])
            
            # Delete high frequencies
            k = 8
            for l in range(7,-1,-1):
                if k > 0:
                    dct_block[l,-k:] = 0
                    k -= 1
            
            dct[i*mblock_size:(i+1)*mblock_size,j*mblock_size:(j+1)*mblock_size] = dct_block
    
    return dct

def homographic_transform(img_src,size,pts):
    [pts_src,pts_dst] = pts
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(img_src, h, size)
    return im_out
    