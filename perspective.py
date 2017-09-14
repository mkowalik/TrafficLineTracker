import numpy as np

class PerspectiveRemover:

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

    def process(self, img):

        self.out_height = self.out_height_factor * img.shape[0]
        self.out_width = self.out_width_factor * img.shape[1]

        self.height = img.shape[0]
        self.width = img.shape[1]

        new_shape = list(img.shape)
        new_shape[0] = self.out_height
        new_shape[1] = self.out_width
        new_shape = tuple(new_shape)

        out = np.zeros(new_shape, dtype=np.uint8)

        angle_max = np.pi / 2. - self.beta - self.theta_hat
        gap = self.h * self.h_factor * np.tan(angle_max)
        self.gap = int(gap)

        for y in range(1, self.out_height-1):
            y_eq = y + gap
            v = ((np.arctan(float(y_eq) / (self.h * self.h_factor)) - np.pi/2. + self.beta + self.theta_hat) / (2. * self.beta)) * self.height
            y_index = int(self.out_height - y - 1)
            v_index = self.height - v - 1.

            if int(v_index) < 0 or int(v_index) + 1 >= self.height:
                continue

            for x in range(1, self.out_width-1):

                u = ((np.arctan(float(x - self.width/2)/y_eq) + self.alpha) / (2. * self.alpha)) * self.width

                if int(v_index) >= 0 and int(v_index)+1 < self.height and int(u) >= 0 and int(u)+1 < self.width:
                    val = 0.0
                    val += img[int(v_index)+1][int(u)] * (v_index%1.0) * (u%1.0)
                    val += img[int(v_index)+1][int(u)+1] * (v_index % 1.0) * (1. - (u % 1.0))
                    val += img[int(v_index)][int(u)] * (1.-(v_index % 1.0)) * (u%1.0)
                    val += img[int(v_index)][int(u)+1] * (1.-(v_index % 1.0)) * (1. - (u % 1.0))
                    out[y_index][x] = int(val)

        return out

    def processReverse(self, img):

        if img.shape[0] != self.out_height or img.shape[1] != self.out_width:
            raise RuntimeWarning("Wrong shape of image")

        out = np.zeros((self.height, self.width), dtype=np.uint8)

        for v in range(self.height):

            v_index = self.height - 1 - int(v)

            angle = np.pi/2. - self.beta - self.theta_hat + (float(v)/float(self.height)) * 2.0 * self.beta
            y = self.h * self.h_factor * np.tan(angle)

            y_index = int(self.out_height - y) + self.gap

            if y_index < 0:
                break

            if y_index >= self.out_height:
                continue

            for u in range(self.width):
                x = y * np.tan(-self.alpha + (float(u)/self.width) * 2.0 * self.alpha) + self.width/2

                if y_index >= 0 and y_index < self.out_height and int(x) >=0 and int(x) < self.out_width:
                    out[v_index][u] = img[y_index][int(x)]

        return out