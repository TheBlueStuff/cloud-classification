import cv2
import numpy as np


########### FUNCTIONS ###########

def readImage(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) #C,H,W
    return image