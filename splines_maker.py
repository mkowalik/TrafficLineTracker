import numpy as np
import cv2

class SplinesMaker:

    def __init__(self, config):

        self.lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')

    def __dist(self, a, b):
        return np.linalg.norm(a-b)

    def __mark_splines(self, y, x):

        for v in range(self.lineWidth):
            for u in range(self.lineWidth):
                self.splines_markers[max(y - v, 0)]                              [max(x - u, 0)]                               = 1
                self.splines_markers[max(y - v, 0)]                              [min(x + u, self.splines_markers.shape[1]-1)] = 1
                self.splines_markers[min(y + v, self.splines_markers.shape[0]-1)][max(x - u, 0)]                               = 1
                self.splines_markers[min(y + v, self.splines_markers.shape[0]-1)][min(x + u, self.splines_markers.shape[1]-1)] = 1


    def prepare_max_list(self, img, max_value=255):

        out = np.zeros_like(img)
        self.splines_markers = np.zeros_like(img)

        self.max_list = []

        for y, row in enumerate(img):
            for x, e in enumerate(row):

                if e>=max_value and row[x+(self.lineWidth/2)]==255:

                    if self.splines_markers[y][x] == 0:
                        self.max_list.append((y, x + (self.lineWidth / 2)))
                        out[y][x + (self.lineWidth / 2)] = 1
                        self.__mark_splines(y, x + (self.lineWidth / 2))

        # out *= 255 # TODO debug

        return out

    def __count_sobel(self, img, y, x, r):
        cups = np.zeros(9, dtype=np.float64)

        ry_max = min(y+r, img.shape[0])
        ry_min = max(0, y-r)

        rx_max = min(x+r, img.shape[1])
        rx_min = max(0, x-r)

        for v in range(ry_min, ry_max):
            for u in range(rx_min, rx_max):
                i = int(self.angle[v][u])
                cups[i] += self.mag[v][u]

        return np.argmax(cups)

    def compute_direction(self, img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # TODO mozna poprobowac inne ksize
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        self.mag, self.angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        self.angle = self.angle % 180.
        self.angle = (self.angle / 20.)
        self.angle = np.uint8(self.angle)

        self.max_angle_index = []

        for (y, x) in self.max_list:
            val = self.__count_sobel(img, y, x, 2 * self.lineWidth)
            self.max_angle_index.append(val)

    def get_visualisation_of_directions(self, img):
        orientations = 9.
        sy, sx = img.shape
        cx, cy = (8, 8)

        hog_im = np.zeros_like(img)

        radius = self.lineWidth  # 2 - 1
        orientations_arr = np.arange(orientations)
        dx_arr = radius * np.cos((orientations_arr / orientations) * np.pi)
        dy_arr = radius * np.sin((orientations_arr / orientations) * np.pi)
        for (y, x), s in zip(self.max_list, self.max_angle_index):
            centre = tuple([y, x])
            cv2.line(hog_im, (int(x + dx_arr[s]), int(y + dy_arr[s])),
                     (int(x - dx_arr[s]), int(y - dy_arr[s])), 255)

        # for s in range(9):    # All possible directions
        #     y = s * 20
        #     x = 20
        #     cv2.line(hog_im, (int(x + dx_arr[s]), int(y + dy_arr[s])),
        #              (int(x - dx_arr[s]), int(y - dy_arr[s])), 255)

        return hog_im

    def connect_nearby(self, img):
        pass

    def make_splines(self, img):

        self.compute_direction(img)
        self.connect_nearby(img)

