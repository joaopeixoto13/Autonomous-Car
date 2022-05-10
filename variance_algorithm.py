from cmath import isnan, pi
import cv2 as cv
import numpy as np
import math

WIDTH = 640
HEIGHT = 480
CENTER_COORD = (WIDTH//2, HEIGHT-1)

CENTER_COORD_1 = (WIDTH//4, HEIGHT - 1)
CENTER_COORD_2 =(3*WIDTH//4, HEIGHT - 1)

font = cv.FONT_HERSHEY_COMPLEX

def imgProcess(img):
    img = cv.resize(img, (WIDTH,HEIGHT))                                            # Resize the image to the desired size
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)                                        # Convert to grayscale
    img = cv.blur(img,(5,5))                                                        # Blur the image to remove noise
    img = cv.Canny(img,100,200)                                                     # Apply Canny edge detection
    return img

def drawLidar(img, center, start_radius, end_radius):                        
    for i in range (start_radius,end_radius+1,30):                                  # Draw the lidar with 30 degrees between each radius
        cv.ellipse(img,center,(i,i),0,15,165,color = 255, thickness= 1)             # Draw the lidar (ellipse)
    return img

def getContours(img):
    merged_list = []
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)      # Find the contours
    contours = np.asarray(contours, dtype=object)                                   # Convert the contours to an numpy array
    for l in contours:                                                              # For each contour
        for i in l:                                                                 # For each point in the contour
            for j in i:                                                             # For each coordinate in the point
                merged_list.append(j)                                               # Append the coordinate to the list
    return merged_list

def filterExtreme(points):
    x = [i[0] for i in points]                                                      # Get the x coordinates
    y = [i[1] for i in points]                                                      # Get the y coordinates
    x = np.array(x)                                                                 # Convert to numpy array
    y = np.array(y)                                                                 # Convert to numpy array
    x_mean = np.mean(x)                                                             # Get the mean of the x coordinates
    y_mean = np.mean(y)                                                             # Get the mean of the y coordinates
    x_std = np.std(x)                                                               # Get the standard deviation of the x coordinates
    y_std = np.std(y)                                                               # Get the standard deviation of the y coordinates
    x_min = x_mean - 2 * x_std                                                      # Get the minimum x coordinate
    x_max = x_mean + 2 * x_std                                                      # Get the maximum x coordinate
    y_min = y_mean - 2 * y_std                                                      # Get the minimum y coordinate
    y_max = y_mean + 2 * y_std                                                      # Get the maximum y coordinate
    filtered_points = []
    for i in range(0,len(points)):
        if(points[i][0] > x_min and points[i][0] < x_max and\
            points[i][1] > y_min and points[i][1] < y_max):                         # If the point is inside the filter
            filtered_points.append(points[i])                                       # Append the point to the filtered points
    return filtered_points

def varianceControl(pointsl,pointsr):
    global u_array
    xl = [i[0] for i in pointsl]                                                    # Get the left points x coordinates
    xr = [i[0] for i in pointsr]                                                    # Get the right points x coordinates
    if (len(xl) == 0 or len(xr) == 0):                                              # If there are no points
        return 0                                                                    # Return 0                           
    xl_mean = np.mean(xl)                                                           # Get the mean of the left points x coordinates
    xr_mean = np.mean(xr)                                                           # Get the mean of the right points x coordinates
    xl_std = np.std(xl)                                                             # Get the standard deviation of the left points x coordinates
    xr_std = np.std(xr)                                                             # Get the standard deviation of the right points x coordinates
    
    std = (xl_std - xr_std)                                                         # Get the difference between the left and right points x standard deviations
    mean = ((xl_mean + xr_mean + 370) / 2) - 320                                    # Get the mean of the left and right points x coordinates
    u = 0.70*std + 0.30*mean                                                        # Get the control signal (with a weight of 70% of the standard deviation and 30% of the mean)
    u_array = np.roll(u_array, -1, axis=0)                                          # Roll the array to the left
    u_array[4] = u                                                                  # Set the last element of the array to the control signal
    u_array = np.multiply(u_array, n_array)                                         # Multiply the array by the n_array (weighted mean filter)
    u_mean = np.sum(u_array)                                                        # Get the mean of the array
    return u_mean                                                                   # Return the control signal

def controlProcess(u):
    if (u > 60):                                                                    # If the control signal is greater than 60
        u = 60                                                                      # Set the control signal to 60
    elif (u < -60):                                                                 # If the control signal is less than -60
        u = -60                                                                     # Set the control signal to -60
    u = u * Kp                                                                      # Multiply the control signal by the Kp constant
    u = u * -1                                                                      # Multiply the control signal by -1 (Simulator needs the opposite direction)   
    return u                                                                        # Return the control signal

def varianceAlgorithm(img):
    global u
    global u_prev
    global u_array
    global n_array
    u_prev = u
    imgA = imgProcess(img)                                                          # Get the image processed
    imgB = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)                                  # Create a blank image
    drawLidar(imgB, (WIDTH//2,HEIGHT//3), 0, 450)                                   # Draw the lidar
    imgC = cv.bitwise_and(imgA, imgB)                                               # Apply the mask  
    left =  np.asarray(getContours(imgC[:,:WIDTH//2 - 50]), dtype=object)           # Get the left contours
    right =  np.asarray(getContours(imgC[:,WIDTH//2 + 50:WIDTH]), dtype=object)     # Get the right contours
    if len(right) > 0 and len(left) > 0:                                            # If there are contours
        u = varianceControl(filterExtreme(left),filterExtreme(right))               # Get the control signal
    else:                                                                           # If there are no contours
        return 0
    u = controlProcess(u)                                                           # Control the direction
    return u                                                                        # Return the control signal

def varianceClearSignal():                                                          # Clear the control signal                          
    u_array[0] = u_array[1] = u_array[2] = u_array[3] = u_array[4] = 0              # Set the control signal array to 0 

# Variables
n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])                                    # Create the n_array (weighted mean filter)
u_array =  np.array([0,0,0,0,0], dtype=float)                                       # Create the u_array (control signal array)
u = 0                                                                               # Initialize the control signal
u_prev = 0                                                                          # Initialize the previous control signal
Kp = 0.75                                                                           # Set the Kp constant

