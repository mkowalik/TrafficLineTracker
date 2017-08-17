import cv2
import ConfigParser

def test():

    config = ConfigParser.RawConfigParser()
    config.read("properties.properties")

    cameraL = config.getint('Video Section', 'video.CameraL')
    cameraR = config.getint('Video Section', 'video.CameraR')


    for i in range(0, 10):
        capL = cv2.VideoCapture(i)
        if capL.isOpened():
            capLindex = i
            break

    # cv2.VideoCapture.ge
    capL.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    capL.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    print capL.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print capL.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    # for i in range(1, 10):
    #     if i==capLindex:
    #         continue
    #     capR = cv2.VideoCapture(i)
    #     if capR.isOpened():
    #         capRindex = i
    #         break
    #
    # print capLindex, " ", capRindex
    #
    # capR.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 600)
    # capR.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 400)
    #
    # print capR.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    # print capR.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    #
    left = True

    while (True):

        if left:
            retL, frameL = capL.read()
            cv2.imshow('frameL', frameL)
        # else:
        #
        #     print capR.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        #     print capR.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        #
        #     retR, frameR = capR.read()
        #     gray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        #     cv2.imshow('frameR', gray)
        #
        # left = not left

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
