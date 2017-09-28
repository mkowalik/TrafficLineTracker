import numpy as np
from timing import timing
import cv2

class OneDimKernelMerge:

    def __init__(self, config):

        self.lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')
        self.kernelSize = 2*self.lineWidth

    @timing
    def process(self, img):

        out = np.zeros_like(img)

        kernel_left = np.ones((self.lineWidth*2))
        kernel_right = np.ones((self.lineWidth * 2))

        for i in range(self.lineWidth/2):
            kernel_left[i] = -2.
            kernel_left[self.lineWidth*2-1-i] = 0.
            kernel_left[self.lineWidth/2+1+i] = 1.
            kernel_left[self.lineWidth+i] = 1.    # TODO czy to dobrze?

            kernel_right[i] = 0.
            kernel_right[self.lineWidth*2-1-i] = -2
            kernel_right[self.lineWidth/2+1+i] = 1.
            kernel_right[self.lineWidth+i] = 1.    # TODO czy to dobrze?

        test = kernel_left.reshape(1, -1)

        out_left = cv2.filter2D(img, -1, test)
        out_right = cv2.filter2D(img, -1, kernel_right.reshape(1, -1))
        out = np.minimum(out_left, out_right)

        return out


