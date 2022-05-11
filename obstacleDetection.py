import math
import cv2 as cv
import numpy as np
from utils import filterGreen

WIDTH = 640
HEIGHT = 480 
BLOCK_NUMBER = 15000                                                            # Number of points to be considered
BLOCK_X_TH = WIDTH//2                                                           # X center coordinate of the block
BLOCK_Y_TH = HEIGHT//3                                                          # Y center coordinate of the block
BLOCK_X_TH_TH = 40                                                              # Threshold for the x coordinate of the block
 
def detectObstacle(img, angle):
    block = cv.resize(img, (WIDTH,HEIGHT))                                      # Resize the image
    block = filterGreen(block)                                                  # Filter the image
    gray_block = cv.cvtColor(block, cv.COLOR_BGR2GRAY)                          # Convert the image to grayscale
    _, gray_block = cv.threshold(gray_block, 50, 255, cv.THRESH_BINARY)         # Threshold the image
    pos = np.nonzero(gray_block)                                                # Get the non zero coordinates

    if len(pos[0]) < BLOCK_NUMBER:                                              # If there are no points
        return 0                                                                # Return 0

    xmin = pos[1].min()                                                         # Get the minimum x coordinate
    xmax = pos[1].max()                                                         # Get the maximum x coordinate
    ymin = pos[0].min()                                                         # Get the minimum y coordinate 
    ymax = pos[0].max()                                                         # Get the maximum y coordinate

    print(f"N: {len(pos[0])}, Angle: {angle}, xmin: {xmin}, xmax: {xmax}, ymax: {ymax}")
    
    if len(pos[0]) > BLOCK_NUMBER and ymax >= BLOCK_Y_TH and\
     (math.fabs((xmax+xmin)//2 - (BLOCK_X_TH - angle*15)) < BLOCK_X_TH_TH):  
        return 1                                                                # Return 1
    else:                                                                       # If there are less than 8000 points or the block is not in the middle of the image
        return 0                                                                # Return 0