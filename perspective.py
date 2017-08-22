
import numpy as np


class PerspectiveRemover:

    """
    Args:
        theta: angle from top view to horizon line in radians
        alpha: vertical angle in radians of camera visible range
        beta: horizontal nagle in radians of camera visible range
    """
    def __init__(self, config, height, width, out_height, out_width):

        self.h = config.getfloat('Single Image Processing Section', 'image.h')
        self.h_factor = config.getfloat('Single Image Processing Section', 'image.h_factor')
        self.theta = (config.getfloat('Single Image Processing Section', 'image.theta_deg') / 360.) * 2. * np.pi
        self.alpha = (config.getfloat('Single Image Processing Section', 'image.alpha_deg') / 360.) * 2. * np.pi
        self.beta = (config.getfloat('Single Image Processing Section', 'image.beta_deg') / 360.) * 2. * np.pi

        self.height = height
        self.width = width
        self.out_height = out_height
        self.out_width = out_width

        self.theta_hat = self.beta-self.theta

    # def remove_perspective(self, img):
    #
    #     out = np.zeros_like(img)
    #
    #     for v in range(self.height):
    #
    #         angle = np.pi/2. - self.beta - self.theta_hat + (float(v)/self.height)*2.0*self.beta
    #         y = self.h * self.h_factor * np.tan(angle)
    #
    #         y_index = int(self.out_height - y)
    #         # print y_index
    #
    #         for u in range(self.width):
    #             x = y * np.tan(-self.alpha + (float(u)/self.width) * 2.0 * self.alpha) + self.width/2
    #
    #             if y_index < self.out_height and y_index >= 0 and int(x) < self.out_width:
    #                 out[int(y_index)][int(x)] = img[self.height - v-1][u]
    #
    #         # print angle, (angle/6.28) * 350, y
    #
    #     return out

    def transformItoW(self, img):

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

    def __zero_rectangle(self, img, x1, x2, y1, y2, value):

        for y in range(min(y1, y2), max(y1, y2)):
            for x in range(min(x1, x2), max(x1, x2)):
                img[y][x] = value


    def getBorderMask(self, img, border_width, default_value=1, border_value=0):

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
                self.__zero_rectangle(out, x1, x1_prev, y_index, y_index - border_width, border_value)
                self.__zero_rectangle(out, x2, x2_prev, y_index, y_index - border_width, border_value)
            else:
                self.__zero_rectangle(out, x1, x2, y_index, y_index - border_width, border_value)

            x1_prev = x1
            x2_prev = x2

        return out




