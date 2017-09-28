import numpy as np
import cv2
from matplotlib import pyplot as plt
from timing import timing

class AdaptiveThreshold:

    def __init__(self, config):

        self.k = config.getfloat('Adaptive Threshold Section', 'threshold.k')

    @timing
    def process(self, img):

        max_val = np.max(img)

        t = float(max_val) / self.k

        test = 255 - img
        th, out = cv2.threshold(test, 0, 250, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = 255-th
        out = 255 - out

        # t = 37
        # _, out = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        _, out = cv2.threshold(out, t, 255, cv2.THRESH_BINARY)
        print th, t

        return out