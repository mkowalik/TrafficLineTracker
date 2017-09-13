import numpy as np
from morphological import MorphologicalOperations

class NoPerspectiveSideEdgesEraser:

    """
    Args:
        theta: angle from top view to horizon line in radians
        alpha: vertical angle in radians of camera visible range
        beta: horizontal nagle in radians of camera visible range
    """
    def __init__(self, config):

        self.h = config.getfloat('Single Image Processing Section', 'image.h')
        self.h_factor = config.getfloat('Single Image Processing Section', 'image.h_factor')
        self.theta = (config.getfloat('Single Image Processing Section', 'image.theta_deg') / 360.) * 2. * np.pi
        self.alpha = (config.getfloat('Single Image Processing Section', 'image.alpha_deg') / 360.) * 2. * np.pi
        self.beta = (config.getfloat('Single Image Processing Section', 'image.beta_deg') / 360.) * 2. * np.pi

        self.out_height_factor = config.getint('Single Image Processing Section', 'image.out_height_factor')
        self.out_width_factor = config.getint('Single Image Processing Section', 'image.out_width_factor')

        self.theta_hat = self.beta-self.theta

        self.border_width = config.getint('Side Edges Eraser Section', 'side_edges_eraser.border_width')

        self.mo = MorphologicalOperations(config)

    def __zero_rectangle(self, img, x1, x2, y1, y2, value):

        for y in range(min(y1, y2), max(y1, y2)):
            for x in range(min(x1, x2), max(x1, x2)):
                img[y][x] = value

    def __process_morphological_operations(self, img):
        return self.mo.erode(img, iterations=6)

    def process(self, img, default_value=1, border_value=0):

        self.out_height = img.shape[0]
        self.out_width = img.shape[1]

        self.height = img.shape[0] / self.out_height_factor
        self.width = img.shape[1] / self.out_width_factor

        angle_max = np.pi / 2. - self.beta - self.theta_hat
        gap = self.h * self.h_factor * np.tan(angle_max)
        self.gap = int(gap)

        out = np.ones_like(img) * default_value

        u1 = 0
        u2 = self.width-1

        x1_prev = None
        x2_prev = None

        for v in range(self.height):
            angle = np.pi / 2. - self.beta - self.theta_hat + (float(v) / self.height) * 2.0 * self.beta
            y_eq = self.h * self.h_factor * np.tan(angle)
            y = y_eq - self.gap
            y_index = int(self.out_height - y)

            x1 = y_eq * np.tan(-self.alpha + (float(u1) / self.width) * 2.0 * self.alpha) + self.width / 2
            x2 = y_eq * np.tan(-self.alpha + (float(u2) / self.width) * 2.0 * self.alpha) + self.width / 2

            x1 = int(x1)
            x2 = int(x2)

            if y_index>=self.out_height or x1<0 or x2>self.out_width:
                break

            if x1_prev!=None and x2_prev!=None:
                self.__zero_rectangle(out, x1, x1_prev, y_index, y_index - self.border_width, border_value)
                self.__zero_rectangle(out, x2, x2_prev, y_index, y_index - self.border_width, border_value)
            else:
                self.__zero_rectangle(out, x1, x2, y_index, y_index - self.border_width, border_value)

            x1_prev = x1
            x2_prev = x2

        out = self.__process_morphological_operations(out)

        return out * img
