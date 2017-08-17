import pygame.camera
import pygame.image
import pygame

import pygame.camera

# PYGAME NIE WSPIERA WINDOWSA :-(

def pygameTest():

    pygame.camera.init()

    cam = pygame.camera.Camera(0, (640, 480), "RGB")

    cam.start()

    img = pygame.Surface((640, 480))

    cam.get_image(img)

    pygame.image.save(img, "img2.jpg")

    cam.stop()