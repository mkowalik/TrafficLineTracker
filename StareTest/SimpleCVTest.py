from SimpleCV import *
import time
import threading

cam0 = None
cam1 = None
bufThread = None

def testSimpleCV():
    cam0 = Camera(0, prop_set={"width": 320, "height": 240, "delay": 1})
    cam1 = Camera(2, prop_set={"width": 320, "height": 240, "delay": 1})

    disp = Display()

    while disp.isNotDone():
        img1 = cam0.getImage()
        img2 = cam1.getImage()
        if disp.mouseLeft:
            break
        imgOrg = img1.sideBySide(img2)

        imgOut = imgOrg.edges(t1=180)
        imgOrg.sideBySide(imgOut, 'bottom').show()
        time.sleep(0.2)