import cv2
from perspective import PerspectiveRemover
from kernel import OneDimKernel
import numpy as np
import re
from adaptive_threshold import AdaptiveThreshold
from splines_maker import SplinesMaker

def process(config):

    lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')

    path = config.get('Single Image Processing Section', 'image.path')
    prefix = config.get('Single Image Processing Section', 'image.prefix')
    full_path = prefix+path

    img = cv2.imread(full_path, 0)

    pr = PerspectiveRemover(config, img.shape[0], img.shape[1], img.shape[0] * 2, img.shape[1])
    k  = OneDimKernel(config)
    at = AdaptiveThreshold(config)
    sm = SplinesMaker(config)

    removed_perspective_image = pr.transformItoW(img)
    # kernelised_huang = k.kerneliseGOLD(removed_perspective_image)
    kernelised_huang = k.kerneliseHuang(removed_perspective_image)
    dil = at.proceessMorphological(kernelised_huang)
    thre = at.processThreshold(dil)

    border_mask = pr.getBorderMask(thre, lineWidth, 1, 0)
    border_mask_dil = cv2.erode(border_mask, kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), iterations=6)

    thre = thre*border_mask_dil

    # for v in range(border_mask_dil.shape[0]):
    #     for u in range(border_mask_dil.shape[1]):
    #         if (border_mask_dil[v][u]):
    #             thre[v][u] = 127

    max_img = sm.prepare_max_list(thre)

    sm.make_splines(thre)  #TODO mozna zrobic test jak dziala thre a jak dziala dil

    # cv2.imshow("imgHuang", kernelised_huang)
    cv2.imshow("max", max_img*255)
    cv2.imshow("before", removed_perspective_image)
    cv2.imshow("dil", dil)
    cv2.imshow("thre", thre)
    cv2.imshow("directions", sm.visualise_directions(thre))
    cv2.imshow('splines', sm.visualise_splines_models(thre))

    m = re.split(".jpg", full_path)
    cv2.imwrite(m[0] + '_noPersp.jpg', removed_perspective_image)
    cv2.imwrite(m[0] + '_filtered.jpg', kernelised_huang)
    cv2.imwrite(m[0] + '_dil.jpg', dil)
    cv2.imwrite(m[0] + '_threshold.jpg', thre)
    cv2.imwrite(m[0] + '_max.jpg', max_img*255)
    cv2.imwrite(m[0] + '_splines.jpg', sm.visualise_splines_models(thre))

    cv2.waitKey(0)
    cv2.destroyAllWindows()