import numpy as np
import cv2
import pickle
"""
a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
unique, counts = np.unique(a, return_counts=True)
reps = dict(zip(unique, counts))

print(reps[0], reps[1])
"""

with open("/home/david/Desktop/M1/Team4/week2/result.pkl", 'rb') as f:
    data = pickle.load(f)

print(data)