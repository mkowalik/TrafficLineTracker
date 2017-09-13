import cv2
import numpy as np
import re
import time

from perspective            import PerspectiveRemover
from one_dim_kernel_gold    import OneDimKernelGOLD
from adaptive_threshold     import AdaptiveThreshold
from splines_maker          import SplinesMaker
from one_dim_kernel_gold    import OneDimKernelGOLD
from one_dim_kernel_huang   import OneDimKernelHuang
from morphological          import MorphologicalOperations
from side_edges_eraser      import NoPerspectiveSideEdgesEraser

def process(config):

    lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')

    path = config.get('Single Image Processing Section', 'image.path')
    prefix = config.get('Single Image Processing Section', 'image.prefix')
    full_path = prefix+path

    img = cv2.imread(full_path, 0)

    pr  = PerspectiveRemover(config)
    kg  = OneDimKernelGOLD(config)
    kh  = OneDimKernelHuang(config)
    at  = AdaptiveThreshold(config)
    see = NoPerspectiveSideEdgesEraser(config)
    mo  = MorphologicalOperations(config)
    sm  = SplinesMaker(config)

    removed_perspective_image = pr.process(img)
    # kernelised_huang =        kg.process(removed_perspective_image)
    kernelised_huang =          kh.process(removed_perspective_image)
    dil =                       mo.proceess(kernelised_huang)
    thre =                      at.process(dil)
    no_side_edge =              see.process(thre)
    splines_visualisation =     sm.process(no_side_edge)  #TODO mozna zrobic test jak dziala thre a jak dziala dil

    # cv2.imshow("imgHuang", kernelised_huang)
    cv2.imshow("max", sm.visualise_max_image(thre)*255)
    cv2.imshow("before", removed_perspective_image)
    cv2.imshow("dil", dil)
    cv2.imshow("thre", thre)
    cv2.imshow("directions", sm.visualise_directions(thre))
    cv2.imshow('splines', splines_visualisation)

    m = re.split(".jpg", full_path)
    cv2.imwrite(m[0] + '_noPersp.jpg', removed_perspective_image)
    cv2.imwrite(m[0] + '_filtered.jpg', kernelised_huang)
    cv2.imwrite(m[0] + '_dil.jpg', dil)
    cv2.imwrite(m[0] + '_threshold.jpg', thre)
    cv2.imwrite(m[0] + '_max.jpg', sm.visualise_max_image(thre)*255)
    cv2.imwrite(m[0] + '_splines.jpg', sm.visualise_splines_models(thre))
    cv2.imwrite(m[0] + '_directions.jpg', sm.visualise_directions(thre))

    cv2.waitKey(0)
    cv2.destroyAllWindows()