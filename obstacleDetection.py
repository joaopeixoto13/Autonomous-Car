import math
import cv2 as cv
import numpy as np

WIDTH = 640
HEIGHT = 480
BLOCK_NUMBER = 8000
BLOCK_X_TH = WIDTH//2
BLOCK_Y_TH = HEIGHT//3
BLOCK_X_TH_TH = 80

def filterBlock(img):
    lower = np.array([0, 150, 0], dtype = "uint8")
    upper = np.array([100, 255, 100], dtype = "uint8")
    mask = cv.inRange(img, lower, upper)
    mask = cv.bitwise_and(img, img, mask=mask)
    return mask
 
def detectObstacle(img):
    block = cv.resize(img, (WIDTH,HEIGHT))
    block = filterBlock(block)
    gray_block = cv.cvtColor(block, cv.COLOR_BGR2GRAY)
    _, gray_block = cv.threshold(gray_block, 50, 255, cv.THRESH_BINARY)
    pos = np.nonzero(gray_block)

    # If there is no obstacle, return false
    if len(pos) == 0:
        return 0

    xmin = pos[1].min()
    xmax = pos[1].max()
    ymin = pos[0].min()
    ymax = pos[0].max()

    print(len(pos[0]))
    if len(pos[0]) > BLOCK_NUMBER and ymax >= BLOCK_Y_TH and (((xmax+xmin)//2 - BLOCK_X_TH) < BLOCK_X_TH_TH):
        return 1
    else:
        return 0