import numpy as np
import cv2

class AdaptiveThreshold:

    def __init__(self, config):

        self.k = config.getfloat('Adaptive Threshold Section', 'threshold.k')
        self.c = config.getint('Adaptive Threshold Section', 'threshold.c')

    def process(self, img):

        out = np.zeros_like(img)

        max_val = None

        for y, row in enumerate(img):

            for x, e in enumerate(row):

                if y % (self.c/2) == 0 and x % (self.c/2) == 0:
                    y_min = max(0, y)
                    y_max = min(img.shape[0], y+self.c)
                    x_min = max(0, x)
                    x_max = min(img.shape[1], x+self.c)

                    test = img[y_min:y_max,x_min:x_max].ravel()

                    max_val = np.max(test)

                t = float(max_val)/self.k
                if (e > t):
                    out[y][x] = 255
                else:
                    out[y][x] = 0

        return out