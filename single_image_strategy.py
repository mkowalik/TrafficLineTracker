import cv2
import numpy as np
import re
import time

from perspective            import PerspectiveRemover
from adaptive_threshold     import AdaptiveThreshold
from splines_maker          import SplinesMaker
from one_dim_kernel_gold    import OneDimKernelGOLD
from one_dim_kernel_huang   import OneDimKernelHuang
from one_dim_kernel_merge   import OneDimKernelMerge
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
    # kh  = OneDimKernelHuang(config)
    km  = OneDimKernelMerge(config)
    at  = AdaptiveThreshold(config)
    see = NoPerspectiveSideEdgesEraser(config)
    mo  = MorphologicalOperations(config)
    sm  = SplinesMaker(config)
    ic  = ImagesConverter()

    print "And go..."

    removed_perspective =       pr.process(img)
    print "Perspective done"
    # kernelised_GOLD =         kg.process(removed_perspective)
    # kernelised_huang =          kh.process(removed_perspective)
    kernelised_merge =          km.process(removed_perspective)
    print "One-dim kernel done"
    dil =                       mo.proceess(kernelised_merge)
    print "Dilation 1 done"
    thresholded =               at.process(dil)
    print "Threshold done"
    no_side_edge =              see.process(thresholded)
    print "No-side done"
    thresholded_dil =           mo.proceess(no_side_edge)
    print "Threshold 2 done"
    _ =                         sm.process(thresholded_dil)  #TODO mozna zrobic test jak dziala thresholded a jak dziala dil
    print "Splines done"

    splines_visualisation               = sm.visualise_splines_models(thresholded, lineWidth/2, 255)
    splines_visualisation_perspective   = pr.processReverse(splines_visualisation)
    line_mask                           = ic.convert_to_one_color(splines_visualisation_perspective, (0, 0, 255))
    merged_with_mask_perspective               = ic.merge_with_mask(imgColor, line_mask)

    # # cv2.imshow("imgHuang", kernelised_huang)
    cv2.imshow("removed_perspective", removed_perspective)
    cv2.imshow("kernelised_huang", kernelised_merge)
    cv2.imshow("dil", dil)
    cv2.imshow("thresholded", thresholded)
    cv2.imshow("thresholded_dil", thresholded_dil)
    cv2.imshow("max", sm.visualise_max_image(thresholded))
    cv2.imshow("directions", sm.visualise_directions(thresholded))
    cv2.imshow("splines", sm.visualise_splines_models(thresholded, lineWidth/2, 255))

    cv2.imshow("merged_with_mask_perspective", merged_with_mask_perspective)

    m = re.split(".jpg", full_path)
    cv2.imwrite(m[0] + '_removed_perspective.jpg', removed_perspective)
    cv2.imwrite(m[0] + '_kernelised_merge.jpg', kernelised_merge)
    cv2.imwrite(m[0] + '_dil.jpg', dil)
    cv2.imwrite(m[0] + '_thresholded.jpg', thresholded)
    cv2.imwrite(m[0] + '_thresholded_dil.jpg', thresholded_dil)
    cv2.imwrite(m[0] + '_max.jpg', sm.visualise_max_image(thresholded))
    cv2.imwrite(m[0] + '_splines.jpg', sm.visualise_splines_models(thresholded, lineWidth/2, 255))
    cv2.imwrite(m[0] + '_directions.jpg', sm.visualise_directions(thresholded))

    cv2.imwrite(m[0] + '_merged_with_mask_perspective.jpg', merged_with_mask_perspective)

    cv2.waitKey(0)
    cv2.destroyAllWindows()