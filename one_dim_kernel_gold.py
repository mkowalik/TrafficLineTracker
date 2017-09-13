import numpy as np

class OneDimKernelGOLD:

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

                l = int(row[x])
                c = int(row[x+self.lineWidth])
                r = int(row[x+2*self.lineWidth])

                d_p = c - l
                d_m = c - r

                if (d_p>0 and d_m>0):
                    out[y][x+self.lineWidth] = int(d_p+d_m)
                else:
                    out[y][x+self.lineWidth] = 0

        return out
