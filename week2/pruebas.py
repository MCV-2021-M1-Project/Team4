import cv2
import numpy as np
from background_substraction import substractBackground

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

numImages = 30

args = {}
args = {'q':'../../data/qsd2_w2','m':'d'}
args = dotdict(args)

substractBackground(numImages,args)


