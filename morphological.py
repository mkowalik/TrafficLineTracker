import numpy as np
import cv2
from timing import timing

class MorphologicalOperations:

    def __init__(self, config):
        self.dilation_iterations = config.getint('Morphological Operations Section', 'threshold.dilation_iterations')
        self.opening_iterations = config.getint('Morphological Operations Section', 'threshold.opening_iterations')

    @timing
    def proceess(self, img, kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)):

        out = np.copy(img)

        out = cv2.dilate(out, kernel, iterations=self.dilation_iterations)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=self.opening_iterations)

        return out

    def dilation(self, img, kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), iterations=None):

        if iterations is None:
            iterations = self.dilation_iterations

        out = np.copy(img)
        out = cv2.dilate(out, kernel, iterations=iterations)

        return out

    def opening(self, img, kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), iterations=None):

        if iterations is None:
            iterations = self.opening_iterations

        out = np.copy(img)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=iterations)

        return out

    def erode(self, img, kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), iterations=1):

        out = np.copy(img)
        out = cv2.erode(img, kernel=kernel, iterations=iterations)

        return out
