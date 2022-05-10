import cv2 as cv
import numpy as np
from utils import filterGreen, drawHist

WIDTH = 640
HEIGHT = 480

def detectCrossWalk(img):
    img_green = filterGreen(img)                                # Filter the image with green color
    img = img - img_green                                       # Subtract the image with the green color (remove the block)
    img = cv.resize(img, (WIDTH,HEIGHT))                        # Resize the image
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)                    # Convert the image to gray scale
    img = cv.blur(img,(5,5))                                    # Blur the image
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)          # Threshold the image
    mask = np.zeros((HEIGHT,WIDTH), np.uint8)                   # Create a mask
    mask[HEIGHT//3:HEIGHT,:] = 255                              # Set the mask to the image
    img = cv.bitwise_and(img, img, mask=mask)                   # Apply the mask to the image
    x, y = drawHist(img, 0)                                     # Draw the histogram
    if (x.mean() >= 60):                                        # If the cumulative histogram detect a crosswalk
        return True                                             # Return true, the image is a crosswalk
    else:                                                       # Otherwise
        return False                                            # Return false, the image is not a crosswalk

def controlProcess(u):
    u = u * Kp                                                  # Proportional control
    if (u > 60):                                                # Limit control
        u = 60                                                  # to 60
    elif (u < -60):                                             # Limit control
        u = -60                                                 # to -60
    #u = u * -1                                                 # Invert control (to make it work on simulator)
    return u                                                    # Return control value

def crossWalkControl(img):
    img_green = filterGreen(img)                        # Filter the image with green color
    img = img - img_green                               # Subtract the image with the green color (remove the block)
    img = cv.resize(img, (WIDTH,HEIGHT))                # Resize the image
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)            # Convert the image to gray scale
    img = cv.blur(img,(5,5))                            # Blur the image
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)  # Threshold the image
    mask = np.zeros((HEIGHT,WIDTH), np.uint8)           # Create a mask
    mask[HEIGHT//3:HEIGHT,:] = 255                      # Set the mask to the image
    img = cv.bitwise_and(img, img, mask=mask)           # Apply the mask to the image
    pos = np.nonzero(img)                               # Get the nonzero values
    xmin = pos[1].min()                                 # Get the minimum x value
    xmax = pos[1].max()                                 # Get the maximum x value
                                                        # Calculate the centroid
    Xc = (xmin + xmax) // 2                             # Get the x center

    M = cv.moments(img)                                 # Get the moments
    cX = int(M["m10"] / M["m00"])                       # Get the x mass center

    u = Xc - cX                                         # Centroid - Mass center
    u = controlProcess(u)                               # Control the process
    return u                                            # Return the control value

Kp = 0.20                                               # Proportional control