import cv2
from perspective import PerspectiveRemover
from kernel import OneDimKernel
import numpy as np
import re
from adaptive_threshold import AdaptiveThreshold

def process(config):

    path = config.get('Single Image Processing Section', 'image.path')

    h = config.getfloat('Single Image Processing Section', 'image.h')
    h_factor = config.getfloat('Single Image Processing Section', 'image.h_factor')
    theta = (config.getfloat('Single Image Processing Section', 'image.theta_deg')/360.) * 2.*np.pi
    alpha = (config.getfloat('Single Image Processing Section', 'image.alpha_deg')/360.) * 2.*np.pi
    beta = (config.getfloat('Single Image Processing Section', 'image.beta_deg')/360.) * 2.*np.pi

    img = cv2.imread(path, 0)

    pr = PerspectiveRemover(h, h_factor, theta, alpha, beta, img.shape[0], img.shape[1])
    k  = OneDimKernel(config)

    removed_perspective_image = pr.transformItoW(img, img.shape[0] * 2, img.shape[1])
    # kernelised_gold = k.kerneliseGOLD(removed_perspective_image)
    kernelised_huang = k.kerneliseHuang(removed_perspective_image)

    cv2.imshow("imgHuang", kernelised_huang)

    at = AdaptiveThreshold(config)
    dil = at.proceessMorphological(kernelised_huang)

    cv2.imshow("dil", dil)
    thre = at.processThreshold(dil)

    cv2.imshow("thre", thre)



    cv2.imshow("before", removed_perspective_image)

    m = re.split(".jpg", path)
    cv2.imwrite(m[0] + '_noPersp.jpg', removed_perspective_image)
    cv2.imwrite(m[0] + '_filtered.jpg', kernelised_huang)
    cv2.imwrite(m[0] + '_dil.jpg', dil)
    cv2.imwrite(m[0] + '_threshold.jpg', thre)

    cv2.waitKey(0)
    cv2.destroyAllWindows()