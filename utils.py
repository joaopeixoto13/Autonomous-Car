from cmath import isnan, pi
import cv2 as cv
from cv2 import determinant
import numpy as np
import math
import imutils

WIDTH = 640
HEIGHT = 480

def filterGreen(img):
    lower = np.array([0, 110, 0], dtype = "uint8")              # Lower bound of green color
    upper = np.array([100, 255, 100], dtype = "uint8")          # Upper bound of green color
    mask = cv.inRange(img, lower, upper)                        # Create a mask with the boundarie
    mask = cv.bitwise_and(img, img, mask=mask)                  # Apply the mask to the image
    return mask                                                 # Return the filtered image

def detectCrossWalk(img):
    img_green = filterGreen(img)                        # Filter the image with green color
    img = img - img_green                               # Subtract the image with the green color (remove the block)
    img = cv.resize(img, (WIDTH,HEIGHT))                # Resize the image
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)            # Convert the image to gray scale
    img = cv.blur(img,(5,5))                            # Blur the image
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)  # Threshold the image
    mask = np.zeros((HEIGHT,WIDTH), np.uint8)           # Create a mask
    mask[HEIGHT//3:HEIGHT,:] = 255                      # Set the mask to the image
    img = cv.bitwise_and(img, img, mask=mask)           # Apply the mask to the image
    x, y = drawHist(img, 0)                             # Draw the histogram
    #print(f"x_mean: {x.mean()}")
    if (x.mean() >= 60):                                # If the cumulative histogram detect a crosswalk
        return True                                     # Return true, the image is a crosswalk
    else:                                               # Otherwise
        return False                                    # Return false, the image is not a crosswalk

def controlProcess(u):
    u = u * Kp                                    # Proportional control
    if (u > 60):                                  # Limit control
        u = 60                                    # to 60
    elif (u < -60):                               # Limit control
        u = -60                                   # to -60
    #u = u * -1                                    # Invert control (to make it work on simulator)
    return u                                      # Return control value

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

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable: 
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def warpingFrame(img):
    WarpingPoints = np.float32([[150, 150], [WIDTH-150, 150],
                       [0, HEIGHT], [WIDTH, HEIGHT]])
    ImagePoints = np.float32([[0, 0], [WIDTH, 0],
                       [0, HEIGHT], [WIDTH, HEIGHT]])

    matrix = cv.getPerspectiveTransform(WarpingPoints, ImagePoints)
    img = cv.warpPerspective(img, matrix, (WIDTH, HEIGHT))
    return img

def drawHist(img, flag):
    img_y_sum = np.sum(img,axis=1)
    img_x_sum = np.sum(img,axis=0)
    img_x_sum = img_x_sum/255
    img_y_sum = img_y_sum/255
    if flag == 1:
        HH = np.zeros((100,img.shape[1]), np.uint8)
        for c in range(img.shape[1]):
            cv.line(HH, (c, 100), (c, 100-int(img_x_sum[c]*100/255)),255)
        cv.imshow('HH', HH)
        HV = np.zeros((img.shape[0],100), np.uint8)
        for l in range(img.shape[0]):
            cv.line(HV, (0, l),(int(img_y_sum[l]*100/255), l), 255)
        cv.imshow('HV', HV)
    return img_x_sum, img_y_sum

def Skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    while True:
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        temp = cv.subtract(img, open)
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        if cv.countNonZero(img)==0:
            break
    return skel

def pointSelectionFilter(img , grid_size):
    final_points = []
    non_filtered_points = np.nonzero(img)
    filter_image = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    grid = np.zeros(((img.shape[0] // grid_size[0]) + 1,(img.shape[1] // grid_size[1]) + 1),np.uint8)
    for x in range(0, len(non_filtered_points[0])):
        point_x = non_filtered_points[1][x]
        point_y = non_filtered_points[0][x]
        grid_x = point_x // grid_size[1]
        grid_y = point_y // grid_size[0]
        if grid[grid_y][grid_x] == 0:
            final_points.append((point_x,point_y))
            grid[grid_y][grid_x] = 1
            filter_image[point_y][point_x] = 255    
    final_points.sort(key=lambda x: x[0])       
    return filter_image, final_points

def drawLines(img, center, start_angle, end_angle):
    for i in range (start_angle,end_angle+1,8):
        CalcXFinal = center[0] + (WIDTH-350) * math.cos(math.radians(i))
        CalcYFinal = center[1] - (WIDTH-350) * math.sin(math.radians(i))
        cv.line(img,(int(center[0]),int(center[1])),(int(CalcXFinal),int(CalcYFinal)),color = 255, thickness = 1)

def drawVirtualLines(img, center, start_angle, end_angle):
    for i in range (start_angle,end_angle+1,10):    
        CalcXFinal = center[0] + (WIDTH//1.5) * math.cos(math.radians(i))
        CalcYFinal = center[1] - (WIDTH//1.5) * math.sin(math.radians(i))
        cv.line(img,(int(center[0]),int(center[1])),(int(CalcXFinal),int(CalcYFinal)),color = 255, thickness= 1)

def detectLines(points, tolerance):
    points.sort(key=lambda x: x[1])
    lines = []
    for i in range(0, len(points) - 3):
        temp = points[i][1]*points[i+1][0] + points[i+1][1]*points[i+2][0] + points[i+2][1]*points[i][0] - points[i+1][1]*points[i][0] - points[i+2][1]*points[i+1][0] - points[i][1]*points[i+2][0]
        if math.fabs(points[i][1] - points[i+2][1]) < WIDTH//5:
            if math.fabs(temp) < tolerance:
                lines.append((points[i][1],points[i][0],points[i+2][1],points[i+2][0],1))
            else:
                for j in range(0, len(lines) - 1):
                    if lines[j][0] == points[i][0] and lines[j][1] == points[i][1]:
                        lines[j][2] = points[i+2][1]
                        lines[j][3] = points[i+2][0]
                        lines[j][4] = lines[j][4] + 1
    return lines

def rotateImage(img, angle):
    rotated = imutils.rotate_bound(img, angle)
    return rotated


Kp = 0.20