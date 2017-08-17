import cv2
from perspective import PerspectiveRemover
import numpy as np

def process(config):

    path = config.get('Single Image Processing Section', 'image.path')

    # img = cv2.imread(path)

    img = cv2.imread(path)



    print 'grey shape: ', img.shape

    pr = PerspectiveRemover(1.0, 30.0, (10./360.) * 2.*np.pi, (30./360.) * 2.*np.pi, (15./360.) * 2.*np.pi, img.shape[0], img.shape[1])

    removed_perspective_image = pr.transformItoW(img, img.shape[0] * 2, img.shape[1])
    print removed_perspective_image.shape
    
    cv2.imshow("img", removed_perspective_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()