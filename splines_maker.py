import numpy as np
import cv2
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Point:
    y = None
    x = None

    def __init__(self, y, x):
        self.y = y
        self.x = x

class Spline_Candidate:

    def __init__(self):
        self.points_list = []
        self.predictor = []

    def add_point(self, p):
        self.points_list.append(p)

    def length(self):
        return len(self.points_list)

    def get_list(self):
        ret = []
        for p in self.points_list:
            ret.append([p.y, p.x])

        return np.array(ret)

    def save_model(self, predictor):
        self.predictor = predictor

    def get_model(self):
        return self.predictor

class SplinesMaker:

    def __init__(self, config):

        self.lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')
        self.bottom_border = config.getint('Splines Section', 'splines.bottom_border')
        self.annulus_height = config.getint('Splines Section', 'splines.annulus_height')
        self.annulus_width = config.getint('Splines Section', 'splines.annulus_width')
        self.minimum_spline_points = config.getint('Splines Section', 'splines.minimum_spline_points') # TODO mozna z tym poeksperymentowac

    def __mark_splines(self, y, x):

        for v in range(self.lineWidth):
            for u in range(self.lineWidth):
                self.splines_markers[max(y - v, 0)]                              [max(x - u, 0)]                               = 1
                self.splines_markers[max(y - v, 0)]                              [min(x + u, self.splines_markers.shape[1]-1)] = 1
                self.splines_markers[min(y + v, self.splines_markers.shape[0]-1)][max(x - u, 0)]                               = 1
                self.splines_markers[min(y + v, self.splines_markers.shape[0]-1)][min(x + u, self.splines_markers.shape[1]-1)] = 1


    def prepare_max_list(self, img, max_value=255):

        self.max_image = np.zeros_like(img, dtype=np.int)
        self.splines_markers = np.zeros_like(img)

        self.max_list = []

        for y, row in enumerate(img):
            for x, e in enumerate(row):

                if e>=max_value and row[x+(self.lineWidth/2)]==255:

                    if self.splines_markers[y][x] == 0:
                        self.max_list.append((y, x + (self.lineWidth / 2)))
                        self.__mark_splines(y, x + (self.lineWidth / 2))

        self.max_list = np.array(self.max_list)
        self.max_list = self.max_list[self.max_list[:, 1].argsort(kind='mergesort')]
        self.max_list = self.max_list[(self.max_list[:, 0] * (-1)).argsort(kind='mergesort')]

        for i, (y, x) in enumerate(self.max_list):
            self.max_image[y][x] = i

        # out *= 255 # TODO debug

        return np.uint8(self.max_image>0)

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

        self.max_angle_index = np.array(self.max_angle_index)

    def visualise_directions(self, img):
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

    def _check_inside(self, p, bl, br, tl, tr):

        m = np.array([[bl.x, bl.y, 1],
                      [tl.x, tl.y, 1],
                      [p.x,  p.y,  1]])

        if np.linalg.det(m) <= 0:
            return False

        m = np.array([[tl.x, tl.y, 1],
                      [tr.x, tr.y, 1],
                      [p.x, p.y, 1]])

        if np.linalg.det(m) <= 0:
            return False

        m = np.array([[tr.x, tr.y, 1],
                      [br.x, br.y, 1],
                      [p.x, p.y, 1]])

        if np.linalg.det(m) <= 0:
            return False

        m = np.array([[br.x, br.y, 1],
                      [bl.x, bl.y, 1],
                      [p.x, p.y, 1]])

        if np.linalg.det(m) <= 0:
            return False

        return True

    def connect_nearby(self, img):

        self.splines_list = []

        img_height = img.shape[0]

        still_in_game = np.ones(self.max_list.shape[0]).astype(np.bool)

        sin_tab = np.sin((np.arange(0., 9.) / 10.) * np.pi)
        cos_tab = np.cos((np.arange(0., 9.) / 10.) * np.pi)

        for spline_begining_index, spline_begining in enumerate(self.max_list):

            if spline_begining[0] < img_height - self.bottom_border:
                break

            if not still_in_game[spline_begining_index]:
                continue

            b_i = spline_begining_index
            b_y = spline_begining[0]
            b_x = spline_begining[1]
            actual_spline = Spline_Candidate()

            still_in_game[b_i] = False

            while True:

                dw_y = self.annulus_width * sin_tab[self.max_angle_index[b_i]]
                dw_x = self.annulus_width * cos_tab[self.max_angle_index[b_i]]
                dh_y = self.annulus_height * cos_tab[self.max_angle_index[b_i]]
                dh_x = self.annulus_height * sin_tab[self.max_angle_index[b_i]]

                bl = Point(b_y + dw_y, b_x - dw_x)
                br = Point(b_y - dw_y, b_x + dw_x)
                tl = Point(b_y - dh_y + dw_y, b_x - dh_x - dw_x)
                tr = Point(b_y - dh_y - dw_y, b_x - dh_x + dw_x)

                min_y = min(b_y - dh_y + dw_y, b_y - dh_y - dw_y)

                candidate = None
                candidate_i = None

                for p_i, (p_y, p_x) in enumerate(self.max_list[b_i:]):

                    if not still_in_game[b_i + p_i]:
                        continue

                    p = Point(p_y, p_x)

                    if p_y < min_y:
                         break

                    if not self._check_inside(p, bl, br, tl, tr):
                        continue

                    if candidate==None or p_y<candidate.y: # TODO cos madrzjszego
                        candidate = p
                        candidate_i = p_i + b_i

                    still_in_game[b_i + p_i] = False
                    actual_spline.add_point(p)

                if candidate == None:
                    break

                b_y = candidate.y
                b_x = candidate.x

            if (actual_spline.length()>self.minimum_spline_points):
                self.splines_list.append(actual_spline)

    def use_ransac(self, img):  # TODO zrobic zeby dzialalo tez na przerywanej

        for spline in self.splines_list:

            # image_part = self.max_list[np.logical_and(self.max_list[:,1] < spline_beginning_x+100, self.max_list[:,1] > 0)]# spline_beginning_x-100)]
            image_part = spline.get_list()
            ransac = linear_model.RANSACRegressor() # TODO mozna sie parametrami pobawic

            model = make_pipeline(PolynomialFeatures(3), ransac) # TODO moze polynomial features 2 a nie 3
            model.fit(image_part[:,0].reshape(-1, 1), image_part[:, 1])

            spline.save_model(model)

    def visualise_splines_models(self, img):

        out = np.zeros_like(img)
        y_plot = np.linspace(0, img.shape[0], img.shape[0], endpoint=False).astype(np.int)

        for spline in self.splines_list:
            model = spline.get_model()
            x_plot = model.predict(y_plot.reshape(-1, 1)).astype(np.int)

            for y, x in zip(y_plot, x_plot):
                if 0 <= x < img.shape[1]:
                    out[y][x] = 255

        return out

    def make_splines(self, img):

        self.compute_direction(img)
        self.connect_nearby(img)
        self.use_ransac(img)


