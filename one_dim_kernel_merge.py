import numpy as np
from timing import timing

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

        for y, row in enumerate(img):
            for x in range(row.shape[0] - self.kernelSize):
                a_left = row[x:x + self.kernelSize] * kernel_left
                a_right = row[x:x + self.kernelSize] * kernel_right
                s_left = int(np.sum(a_left))/2.
                s_right = int(np.sum(a_right))/2.
                s = min(s_left, s_right)
                out[y][x+self.lineWidth] = max(0,s)

        return out


