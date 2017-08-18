
import numpy as np


class PerspectiveRemover:

    """
    Args:
        theta: angle from top view to horizon line in radians
        alpha: vertical angle in radians of camera visible range
        beta: horizontal nagle in radians of camera visible range
    """
    def __init__(self, h, h_factor, theta, alpha, beta, height, width):
        self.h = float(h)
        self.h_factor = float(h_factor)
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.width = width
        self.height = height

        self.theta_hat = beta-theta

    def remove_perspective(self, img, out_height, out_width):

        # Color images support
        new_shape = list(img.shape)
        new_shape[0] = out_height
        new_shape[1] = out_width
        new_shape = tuple(new_shape)

        out = np.zeros(new_shape, dtype=np.uint8)

        for v in range(self.height):

            angle = np.pi/2. - self.beta - self.theta_hat + (float(v)/self.height)*2.0*self.beta
            y = self.h * self.h_factor * np.tan(angle)

            y_index = int(out_height - y)
            # print y_index

            for u in range(self.width):
                x = y * np.tan(-self.alpha + (float(u)/self.width) * 2.0 * self.alpha) + self.width/2


                if y_index < out_height and y_index >= 0 and int(x) < out_width:
                    out[int(y_index)][int(x)] = img[self.height - v-1][u]

            # print angle, (angle/6.28) * 350, y

        return out

    def transformItoW(self, img, out_height, out_width):

        new_shape = list(img.shape)
        new_shape[0] = out_height
        new_shape[1] = out_width
        new_shape = tuple(new_shape)

        out = np.zeros(new_shape, dtype=np.uint8)

        for y in range(1, out_height):
            v = ((np.arctan(float(y) / (self.h * self.h_factor)) - np.pi/2. + self.beta + self.theta_hat) / (2. * self.beta)) * self.height
            y_index = int(out_height - y - 1)
            v_index = int(self.height - v - 1)

            for x in range(1, out_width):

                u = ((np.arctan(float(x - self.width/2)/y) + self.alpha) / (2. * self.alpha)) * self.width

                if v_index >= 0 and v_index < self.height and u >= 0 and u < self.width:
                    out[y_index][x] = img[v_index][int(u)]

        return out




