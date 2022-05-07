import cv2 as cv
import numpy as np
from utils import filterGreen, drawHist

WIDTH = 640
HEIGHT = 480

#n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])
n_array = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
u_array =  np.array([0,0,0,0,0], dtype=float)                                   # Array of control values
Kp = 0.3                                                                        # Proportional gain
X = 0
Y = 0
u = 0
checkPark = 0
parkStatus = 0
OBSTACLE_TH = 2500
OBSTACLE_TH_LIMIT = 13000
NO_OBSTACLE = 0
OBSTACLE_LEFT = 1
OBSTACLE_RIGHT = 2
OBSTACLE_LEFT_AND_RIGHT = 3
LINE_TH = 50

def controlProcess(u):
    global n_array  
    global u_array
    global Kp 
    u_array = np.roll(u_array, -1, axis=0)                      # Roll array to the left
    u_array[4] = u                                              # Insert new value in array
    u_array = np.multiply(u_array, n_array)                     # Multiply array by n_array
    u_mean = np.sum(u_array)                                    # Sum array
    u = u_mean * Kp                                             # Proportional control
    #Kp = 1 - 0.05 * Kp
    if (u > 60):                                                # Limit control
        u = 60                                                  # to 60
    elif (u < -60):                                             # Limit control
        u = -60                                                 # to -60
    u = u * -1                                                  # Invert control (to make it work on simulator)
    return u                                                    # Return control value

def getMoments(img):
    M = cv.moments(img)                                                                       # Get the moments 
    cX = int(M["m10"] / M["m00"])                                                             # Get the x mass center
    cY = int(M["m01"] / M["m00"])                                                             # Get the y mass center 
    return cX, cY

def processPark(img):
    global checkPark
    global X
    global Y
    global u
    global parkStatus
    img_copy = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
    img_green = filterGreen(img)                                                                        # Filter the image with green color
    img = img - img_green                                                                               # Subtract the image with the green color (remove the block)
    img = cv.resize(img, (WIDTH,HEIGHT))                                                                # Resize the image
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)                                                            # Convert the image to gray scale
    img = cv.blur(img,(5,5))                                                                            # Blur the image
    _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)                                               # Threshold the image

    img = cv.morphologyEx(img, cv.MORPH_DILATE, np.ones((10,10), np.uint8), iterations=2)               # Dilate the image
    img = cv.morphologyEx(img, cv.MORPH_ERODE, np.ones((10,10), np.uint8), iterations=2)                # Erode the image

    img_l = img[:, :WIDTH//2]                                                                           # Get the left half of the image
    img_r = img[:, WIDTH//2:]                                                                           # Get the right half of the image	
    
    n_l = np.count_nonzero(img_l)                                                                       # Count the number of non-zero pixels in the left half of the image
    n_r = np.count_nonzero(img_r)                                                                       # Count the number of non-zero pixels in the right half of the image
    #print(f"n_l: {n_l}, n_r: {n_r}")
    
    cX_l, cY_l = getMoments(img_l)                                                                      # Get the x and y mass center of the left half of the image
    cX_r, cY_r = getMoments(img_r)                                                                      # Get the x and y mass center of the right half of the image
    if (n_l < OBSTACLE_TH_LIMIT and n_r < OBSTACLE_TH_LIMIT and checkPark == 0):                        # If the number of non-zero pixels in the left and right half of the image are lower than the threshold
        checkPark = 1                                                                                   # Set the parking to checked and tell the program that is impossible to park
        parkStatus = OBSTACLE_LEFT_AND_RIGHT                                                            # Set the parking status to no obstacle
        print("Impossible to park: Two obstacles detected") 
        return 0, parkStatus                                                                            # Return                             
    else:
        if (n_l < n_r - OBSTACLE_TH):                                                                   # If the obstacle is on the left
            X = cX_r + WIDTH//2                                                                         # Get the x coordinate that the car should go to 
            Y = cY_r                                                                                    # Get the y coordinate that the car should go to
            if (checkPark == 0):                                                                        # If the parking is not checked
                parkStatus = OBSTACLE_LEFT                                                              # Set the parking status to left
                print("Obstacle on the left")
        elif (n_r < n_l - OBSTACLE_TH):
            X = cX_l                                                                                    # Get the x coordinate that the car should go to
            Y = cY_l                                                                                    # Get the y coordinate that the car should go to
            if (checkPark == 0):                                                                        # If the parking is not checked
                parkStatus = OBSTACLE_RIGHT                                                             # Set the parking status to right
                print("Obstacle on the right")
        else:
            X = cX_r + WIDTH//2                                                                         # Get the x coordinate that the car should go to
            Y = cY_r                                                                                    # Get the y coordinate that the car should go to
            if (checkPark == 0):                                                                        # If the parking is not checked
                parkStatus = NO_OBSTACLE
                print("No obstacle")
        checkPark = 1
    
    if (checkPark == 1):                                                                                # If the parking is checked
        if (n_r < n_l - OBSTACLE_TH):
            X = cX_l                                                                                    # Get the x coordinate that the car should go to
            Y = cY_l                                                                                    # Get the y coordinate that the car should go to
        else:
            X = cX_r + WIDTH//2                                                                         # Get the x coordinate that the car should go to 
            Y = cY_r                                                                                    # Get the y coordinate that the car should go to 
    cv.line(img_copy, (X, 0), (X, HEIGHT), 255, 5)                                                      # Draw the vertical line
    #cv.imshow("img", img_copy)
    #cv.line(img_copy, (0, Y), (WIDTH, Y), 255, 2)                                                      # Draw the horizontal line
    #print(f"X: {X}, Y: {Y}, u: {u}")
    error = X - WIDTH//2 
    u = controlProcess(error)
    return u, parkStatus 