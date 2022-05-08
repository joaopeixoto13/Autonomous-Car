from cmath import isnan, pi
from ctypes.wintypes import HWINSTA
import cv2 as cv
from cv2 import determinant
from cv2 import drawContours
import numpy as np
import math
import imutils
from utils import *
from variance_algorithm import varianceAlgorithm, varianceClearSignal

WIDTH = 640
HEIGHT = 480
CENTER_COORD = (WIDTH//2,HEIGHT-1)
font = cv.FONT_HERSHEY_COMPLEX
 

def imgProcess(img):
    img = cv.resize(img, (WIDTH,HEIGHT))                                                                # Resize image
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)                                                            # Convert to grayscale
    img = cv.blur(img,(5,5))                                                                            # Blur image to remove noise
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)                                                  # Threshold image
    img = cv.morphologyEx(img,cv.MORPH_BLACKHAT,(5,5))                                                  # Apply blackhat filter
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (30,1))                                 # Create horizontal kernel
    horizontal_lines = cv.morphologyEx(img, cv.MORPH_CLOSE, horizontal_kernel, iterations=1)            # Apply horizontal filter in order to remove horizontal lines
    img = img - horizontal_lines                                                                        # Subtract horizontal lines from original image, leaving only the vertical and diagonal lines
    return img                                                                                          # Return image

def drawVirtualCircles(img, center, start_radius, end_radius, thick):
    for i in range (start_radius,end_radius+1,10):
        cv.ellipse(img,center,(i,i),0,15,165, color = 255, thickness= thick)
    return img

def trendLine(points):
    x = [i[0] for i in points]                                  # Create x array
    y = [i[1] for i in points]                                  # Create y array
    A = np.vstack([x, np.ones(len(x))]).T                       # Create A matrix
    m, c = np.linalg.lstsq(A, y,rcond=None)[0]                  # Solve for m and c
    return m,c                                                  # Return the slope (m) and y-intercept (c)

def drawTrendLine(slope, b, img,color):
    for i in range(0,WIDTH):                                    # Iterate over all x pixels in image
        y = slope * i + b                                       # Calculate y value for each x
        cv.line(img,(i,int(y)),(i,int(y)),color,1)              # Draw the corresponding line

def IIRFilter(a,y,x):                                           # IIR Filter function
    y = a * y + (1 - a) * x                                     # IIR Filter equation
    return round(y)                                             # Return filtered value

def MeanFilter(u):                                              # Mean Filter function
    global n_array  
    global u_array 
    u_array = np.roll(u_array, -1, axis=0)                      # Roll array to the left
    u_array[4] = u                                              # Insert new value in array
    u_array = np.multiply(u_array, n_array)                     # Multiply array by n_array
    u_mean = np.sum(u_array)                                    # Sum array
    return u_mean                                               # Return mean value

def controlProcess(u):
    u = u * Kp                                    # Proportional control
    if (u > 60):                                  # Limit control
        u = 60                                    # to 60
    elif (u < -60):                               # Limit control
        u = -60                                   # to -60
    u = u * -1                                    # Invert control (to make it work on simulator)
    return u                                      # Return control value

def detectLines(frame, flag):
    global n_lines
    imgA = imgProcess(frame)                                                                # Process image
    imgB = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)                                          # Create empty image
    #imgB2 = imgA.copy()                                                                    # Create empty image

    if (flag):
        imgA = cv.resize(frame, (WIDTH,HEIGHT))                                             # Resize image
        imgA = cv.cvtColor(imgA,cv.COLOR_RGB2GRAY)                                          # Convert to grayscale
        imgA = cv.blur(imgA,(5,5))                                                          # Blur image to remove noise
        _,imgA = cv.threshold(imgA,170,255,cv.THRESH_BINARY)                                # Threshold image
        drawVirtualCircles(imgB, (WIDTH//2,int(HEIGHT//2.587)), 70, 450, 5)                  # Draw virtual circles
    else:
        imgA = imgProcess(frame) 
        drawVirtualCircles(imgB, (WIDTH//2,int(HEIGHT//2.5)), 70, 450, 2)                    # Draw virtual circles
    #drawVirtualCircles(imgB2, (WIDTH//2,int(HEIGHT//2.5)), 70, 450)                         # Draw virtual circles

    imgC = cv.bitwise_and(imgA, imgB)                                                       # Apply bitwise and to get the points
    lines = cv.HoughLinesP(imgC,cv.HOUGH_PROBABILISTIC, np.pi/180, 10,None, 30,200)         # Apply Hough transform to get lines
    if lines is not None:                                                                   # If lines are detected
        for x in range(0, len(lines)):                                                      # Iterate over all lines
            for x1,y1,x2,y2 in lines[x]:                                                    # Iterate over all points in line
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)                            # Create array of points
                cv.polylines(imgC, [pts], True, 255,thickness=25)                           # Draw line

    imgC = cv.morphologyEx(imgC, cv.MORPH_ERODE, np.ones((17,17),np.uint8))                 # Apply erode filter
    contours, _ = cv.findContours(imgC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)             # Find contours
    
    #cv.imshow("imgC", imgC)                                                                  # Show image
    
    final_contours = []
    for cnt in contours:
        cnt = np.squeeze(cnt)
        cnt = list(cnt)
        cnt.sort(key=lambda x: x[0])
        if len(final_contours) == 0:
            final_contours.append(cnt)
        else:
            m,c = trendLine(cnt)
            m2,c2 = trendLine(final_contours[-1])
            if abs(m-m2) < 0.2 and abs(c-c2) < 150:
                pass
            else:
                final_contours.append(cnt)

    n_lines = len(final_contours)
    #n_lines = IIRFilter(0.4, n_lines, len(contours))                                        # IIR Filter
    #cv.imshow("imgC", imgC)
    #print(n_lines)
    return n_lines                                                                          # Return number of lines

def trendLineAlgorithm(frame, priority, n_lines):
    global u_k  
    global u_sum
    global u_inc
    global u
    frame_copy = frame.copy()                                                                                       # Copy frame
    imgA = imgProcess(frame)                                                                                        # Process image
    imgB = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)                                                                  # Create empty image
    #imgB2 = imgA.copy()                                                                                            # Create empty image

    drawVirtualCircles(imgB, (WIDTH//2,int(HEIGHT//2.5)), 70, 450, 2)                                                # Draw virtual circles
    #drawVirtualCircles(imgB2, (WIDTH//2,int(HEIGHT//2.5)), 70, 450)                                                 # Draw virtual circles

    imgC = cv.bitwise_and(imgA, imgB)                                                                               # Apply bitwise and to get the points

    lines = cv.HoughLinesP(imgC,cv.HOUGH_PROBABILISTIC, np.pi/180, 10, None, 30, 200)                               # Apply Hough transform to get lines
    if lines is not None:                                                                                           # If lines are detected
        for x in range(0, len(lines)):                                                                              # Iterate over all lines
            for x1,y1,x2,y2 in lines[x]:                                                                            # Iterate over all points in line
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)                                                    # Create array of points
                cv.polylines(imgC, [pts], True, 255,thickness=25)                                                   # Draw line

    imgC = cv.morphologyEx(imgC, cv.MORPH_ERODE, np.ones((17,17),np.uint8))                                         # Apply erode filter
    contours, _ = cv.findContours(imgC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)                                     # Find contours

    n_lines = IIRFilter(0.4, n_lines, len(contours))                                                                # IIR Filter

    final_contours = []
    for cnt in contours:
        cnt = np.squeeze(cnt)
        cnt = list(cnt)
        cnt.sort(key=lambda x: x[0])
        if len(final_contours) == 0:
            final_contours.append(cnt)
        else:
            m,c = trendLine(cnt)
            m2,c2 = trendLine(final_contours[-1])
            if abs(m-m2) < 0.2 and abs(c-c2) < 150:
                pass
            else:
                final_contours.append(cnt)

    n_lines = len(final_contours)

    line_points = []                                                                                                # Create empty array

    if(n_lines > 0 and n_lines < 3):                                                                                # If there are less than 3 lines
        for contour in final_contours:                                                                                    # Iterate over all contours
            #line_point = cv.approxPolyDP(contour, 3, False)                                                         # Approximate contour
            line_point = np.squeeze(contour)                                                                     # Remove single dimension
            line_point = list(line_point)                                                                           # Convert to list
            line_point.sort(key=lambda x: x[0])                                                                     # Sort points by x value
            if(len(line_point) > 2):                                                                                # If there are more than 2 points
                m,c = trendLine([[line_point[0][0],line_point[0][1]],[line_point[-1][0],line_point[-1][1]]])        # Get trend line
                line_points.append([line_point[0][0],line_point[0][1],line_point[-1][0],line_point[-1][1], m, c])   # Add points to array
        if (len(line_points) == 0):                                                                                 # If there are no points
            return u                                                                                                # Return control value
        if(len(line_points) == 1):                                                                                  # If there is only one point
            if ((priority == PRIORITY_LEFT or priority == PRIORITY_AHEAD) and line_points[0][0]<WIDTH//2):          # If priority is left or ahead and the line detected is on left side
                u = (90 - line_points[0][5])/line_points[0][4] - WIDTH//2                                           # Calculate control value from left line
                u_k = 0                                                                                             # Reset control value
            elif (priority == PRIORITY_RIGHT and line_points[0][2]>WIDTH//2):                                       # If priority is right and the line detected is on right side
                u = (90 - line_points[0][5])/line_points[0][4] - WIDTH//2                                           # Calculate control value from right line
                u_k = 0                                                                                             # Reset control value
            else:                                                                                                   # If priority is left or ahead and the line detected is on right side or priority is right and the line detected is on left side
                if (priority == PRIORITY_LEFT or priority == PRIORITY_AHEAD):                                       # If priority is left or ahead
                    u_k = u_k - u_inc                                                                               # Decrease control value
                    u = u_k * u_sum                                                                                 # Calculate control value
                else:                                                                                               # If priority is right
                    u_k = u_k + u_inc                                                                               # Increase control value
                    u = u_k * u_sum                                                                                 # Calculate control value
            u = MeanFilter(u)                                                                                       # Apply mean filter
            u  = controlProcess(u)                                                                                  # Apply control process
            varianceClearSignal()                                                                                   # Clear variance signal (usefully for new iterations)
            #print(f"trend u: {u}")
        else:                                                                                                       # If there are 2 or more lines
            u1 = ((90 - line_points[0][5])/line_points[0][4] - WIDTH//2)                                            # Calculate control value from left line
            u2 = ((90 - line_points[1][5])/line_points[1][4] - WIDTH//2)                                            # Calculate control value from right line
            #print(f"u1: {u1}, u2: {u2}")
            if (abs(u1 - u2) > 5000):
                u = 0
            else:
                u = (u1 + u2)/2                                                                                     # Calculate control value                                                                                              
            u = MeanFilter(u)                                                                                       # Apply mean filter
            u  = controlProcess(u)                                                                                  # Apply control process
            u_temp = varianceAlgorithm(frame_copy)                                                                  # Apply variance algorithm for smoothing
            #print(f"trend u: {u}, var u: {u_temp}")                                                                 # Print control values
            u = (0.7*u + 0.3*u_temp) / 2                                                                          # Apply weighted mean filter (65% of weighted trendline algorithm and 35% of variance algorithm)
            u_k = 0                                                                                                 # Reset control value
        #print(u)
        return u                                                                                                    # Return control value
    return 0                                                                                                        # Return 0

def trendLineClearSignal():                                                                                         # Clear trendline signal
    u_array[0] = u_array[1] = u_array[2] = u_array[3] = u_array[4] = 0                                              # Set all values to 0

# Variables 
PRIORITY_AHEAD = 0  
PRIORITY_RIGHT = 1
PRIORITY_LEFT = 2

n_lines = 0                                                                 # Number of lines detected
angle = 0                                                                   # Angle of car
m = 0                                                                       # Slope of line
c = 0                                                                       # Y-intercept of line
#n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])                            # Array of weights for mean filter
n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])
u_array =  np.array([0,0,0,0,0], dtype=float)                               # Array of control values
Kp = 0.20                                                                   # Proportional gain
u = 0                                                                       # Control signal
contours = []                                                               # Contours
u_inc = 7                                                                   # Increment of control value
u_k = 0                                                                     # Control value
u_sum = (1 / Kp)                                                            # Sum of control values