import cv2                      # Imported all available (probably used) libraries
from testfile import test
import ConfigParser
import single_image_strategy

useCamera = None
config = None


def loadGeneralProperties():

    global config, useCamera
    config = ConfigParser.RawConfigParser()
    config.read("properties.properties")

    useCamera = config.getint('General Section', 'general.useCamera')


def main():

    print "Michal Kowalik master's thesis application." # Introduction

    loadGeneralProperties()

    if (useCamera):
        test()
    else:
        single_image_strategy.process(config)

if __name__ == "__main__":
    main()                      # Calling proper main function
