import numpy as np

class OneDimKernelHuang:

    def __init__(self, config):

        self.lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')
        self.kernelSize = 2*self.lineWidth

    def process(self, img):

        out = np.zeros_like(img)

        kernel = np.ones((self.lineWidth*2))

        for i in range(self.lineWidth/2):
            kernel[i] = -1.
            kernel[self.lineWidth*2-1-i] = -1.
            kernel[self.lineWidth/2+1+i] = 1.
            kernel[self.lineWidth+i] = 1.    # TODO czy to dobrze?

        max_s = 0

        for y, row in enumerate(img):
            for x in range(row.shape[0] - self.kernelSize):
                a = row[x:x+self.kernelSize] * kernel
                s = int(np.sum(a))/2
                max_s = max(max_s, s)
                out[y][x+self.lineWidth] = max(0,s)

        return out


