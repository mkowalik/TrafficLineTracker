import numpy as np

class Spline_Candidate:

    def __init__(self):
        self.points_list = []
        self.predictor = None

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

    def get_angles_list(self):
        ret = []
        for p in self.points_list:
            ret.append(p.angle)

        return np.array(ret)
