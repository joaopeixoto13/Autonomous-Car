import math
import cv2 as cv
import numpy as np
from utils import filterGreen

WIDTH = 640
HEIGHT = 480
BLOCK_NUMBER = 6500
BLOCK_X_TH = WIDTH//2
BLOCK_Y_TH = HEIGHT//3.3
BLOCK_X_TH_TH = 100
 
def detectObstacle(img):
    block = cv.resize(img, (WIDTH,HEIGHT))                                  # Resize the image
    block = filterGreen(block)                                              # Filter the image
    gray_block = cv.cvtColor(block, cv.COLOR_BGR2GRAY)                      # Convert the image to grayscale
    _, gray_block = cv.threshold(gray_block, 50, 255, cv.THRESH_BINARY)     # Threshold the image
    pos = np.nonzero(gray_block)                                            # Get the non zero coordinates

    if len(pos[0]) < BLOCK_NUMBER:                                          # If there are no points
        return 0                                                            # Return 0

    xmin = pos[1].min()                                                    # Get the minimum x coordinate
    xmax = pos[1].max()                                                    # Get the maximum x coordinate
    ymin = pos[0].min()                                                    # Get the minimum y coordinate 
    ymax = pos[0].max()                                                    # Get the maximum y coordinate

    #print(len(pos[0]))
    if len(pos[0]) > BLOCK_NUMBER and ymax >= BLOCK_Y_TH and (math.fabs((xmax+xmin)//2 - BLOCK_X_TH) < BLOCK_X_TH_TH):   # If there are more than 8000 points and the block is in the middle of the image
        return 1                                                            # Return 1
    else:                                                                   # If there are less than 8000 points or the block is not in the middle of the image
        return 0                                                            # Return 0