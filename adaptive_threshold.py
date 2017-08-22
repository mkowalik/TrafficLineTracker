import numpy as np
import cv2


class AdaptiveThreshold:

    def __init__(self, config):

        self.k = config.getfloat('Adaptive Threshold Section', 'threshold.k')
        self.dilation_iterations = config.getint('Adaptive Threshold Section', 'threshold.dilation_iterations')
        self.opening_iterations = config.getint('Adaptive Threshold Section', 'threshold.opening_iterations')

    def proceessMorphological(self, img, dilate=1, opening=1, kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)): # TODO move morphological operations to seperate file

        out = np.copy(img)

        if (dilate):
            out = cv2.dilate(out, kernel, iterations=self.dilation_iterations)

        if (opening):
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=self.opening_iterations)

        return out

    def processThreshold(self, img):

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