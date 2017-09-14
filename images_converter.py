import cv2
import numpy as np

class ImagesConverter:

    def convert_grey_to_rgb(self, img):

        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return out

    def convert_to_one_color(self, img, color):

        out = self.convert_grey_to_rgb(img)
        for r, row in enumerate(img):
            for p, pix in enumerate(row):
                if pix>0:
                    out[r][p] = color

        return out

    def is_in_color(self, img):
        if len(img.shape) == 3:
            return True
        return False

    def merge_with_mask(self, img, mask):

        if not self.is_in_color(img):
            img = self.convert_grey_to_rgb(img)

        if not self.is_in_color(mask):
            mask = self.convert_grey_to_rgb(mask)

        out = np.copy(img)
        for y, row in enumerate(out):
            for x, pix in enumerate(row):
                if (mask[y][x]).any() != 0:
                    out[y][x] = mask[y][x]

        return out
