import numpy as np
import cv2

class AdaptiveThreshold:

    def __init__(self, config):

        self.k = config.getfloat('Adaptive Threshold Section', 'threshold.k')

    def process(self, img):

        out = np.zeros_like(img)

        max_val = np.max(img)

        t = float(max_val)/self.k

        for y, row in enumerate(img):
            for x, e in enumerate(row):
                if (e < t):
                    out[y][x] = 0
                else:
                    out[y][x] = 255

        return out