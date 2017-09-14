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
from images_converter       import ImagesConverter

def process(config):

    lineWidth = config.getint('Low Level Kernel Section', 'kernel.width')

    path = config.get('Single Image Processing Section', 'image.path')
    prefix = config.get('Single Image Processing Section', 'image.prefix')
    full_path = prefix+path

    img = cv2.imread(full_path, 0)
    imgColor = cv2.imread(full_path, cv2.IMREAD_COLOR)

    pr  = PerspectiveRemover(config)
    # kg  = OneDimKernelGOLD(config)
    kh  = OneDimKernelHuang(config)
    at  = AdaptiveThreshold(config)
    see = NoPerspectiveSideEdgesEraser(config)
    mo  = MorphologicalOperations(config)
    sm  = SplinesMaker(config)
    ic  = ImagesConverter()

    removed_perspective =       pr.process(img)
    # kernelised_GOLD =         kg.process(removed_perspective)
    kernelised_huang =          kh.process(removed_perspective)
    dil =                       mo.proceess(kernelised_huang)
    thresholded =               at.process(dil)
    no_side_edge =              see.process(thresholded)
    _ =                         sm.process(no_side_edge)  #TODO mozna zrobic test jak dziala thresholded a jak dziala dil

    splines_visualisation               = sm.visualise_splines_models(thresholded, lineWidth/2, 255)
    splines_visualisation_perspective   = pr.processReverse(splines_visualisation)
    line_mask                           = ic.convert_to_one_color(splines_visualisation_perspective, (0, 0, 255))
    merged_with_mask_perspective               = ic.merge_with_mask(imgColor, line_mask)

    # # cv2.imshow("imgHuang", kernelised_huang)
    cv2.imshow("removed_perspective", removed_perspective)
    cv2.imshow("kernelised_huang", kernelised_huang)
    cv2.imshow("dil", dil)
    cv2.imshow("thresholded", thresholded)
    cv2.imshow("max", sm.visualise_max_image(thresholded))
    cv2.imshow("directions", sm.visualise_directions(thresholded))
    cv2.imshow("splines", sm.visualise_splines_models(thresholded, lineWidth/2, 255))

    cv2.imshow("merged_with_mask_perspective", merged_with_mask_perspective)

    m = re.split(".jpg", full_path)
    cv2.imwrite(m[0] + '_removed_perspective.jpg', removed_perspective)
    cv2.imwrite(m[0] + '_kernelised_huang.jpg', kernelised_huang)
    cv2.imwrite(m[0] + '_dil.jpg', dil)
    cv2.imwrite(m[0] + '_thresholded.jpg', thresholded)
    cv2.imwrite(m[0] + '_max.jpg', sm.visualise_max_image(thresholded))
    cv2.imwrite(m[0] + '_splines.jpg', sm.visualise_splines_models(thresholded))
    cv2.imwrite(m[0] + '_directions.jpg', sm.visualise_directions(thresholded))

    cv2.imwrite(m[0] + '_merged_with_mask_perspective.jpg', merged_with_mask_perspective)

    cv2.waitKey(0)
    cv2.destroyAllWindows()