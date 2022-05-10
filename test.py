from cmath import isnan, pi
import cv2 as cv
from cv2 import determinant
import numpy as np
import math
import imutils

WIDTH = 640
HEIGHT = 480
CENTER_COORD = (WIDTH//2,HEIGHT-1)
font = cv.FONT_HERSHEY_COMPLEX

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
        hor= np.hstack(imgArray)
        ver = hor
    return ver

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

def ImgProcess(img):
    img = cv.resize(img, (WIDTH,HEIGHT))
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    img = cv.blur(img,(5,5))
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)
    #img = cv.morphologyEx(img,cv.MORPH_ERODE,(5,5),iterations=3)
    #img = Skeleton(img)
    #img = cv.Canny(img,100,200)
    img = warpingFrame(img)
    return img

def drawVirtualLines(img, center, start_angle, end_angle):
    for i in range (start_angle,end_angle+1,10):
        CalcXFinal = center[0] + (WIDTH//1.5) * math.cos(math.radians(i))
        CalcYFinal = center[1] - (WIDTH//1.5) * math.sin(math.radians(i))
        cv.line(img,(int(center[0]),int(center[1])),(int(CalcXFinal),int(CalcYFinal)),color = 255, thickness= 1)

def drawVirtualCircles(img, center, start_radius, end_radius):
    for i in range (start_radius,end_radius+1,30):
        cv.ellipse(img,center,(i,i),0,15,165,color = 255, thickness= 1)
    return img

def pointSelectionFilter(img , grid_size):
    final_points = []
    non_filtered_points = np.nonzero(img)
    filter_image = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    grid = np.zeros((img.shape[0] // grid_size + 1,img.shape[1] // grid_size + 1),np.uint8)
    for x in range(0, len(non_filtered_points[0])):
        point_x = non_filtered_points[0][x]
        point_y = non_filtered_points[1][x]
        grid_x = point_x // grid_size
        grid_y = point_y // grid_size
        if grid[grid_x][grid_y] == 0:
            final_points.append((point_x,point_y))
            grid[grid_x][grid_y] = 1
            filter_image[point_x][point_y] = 255
                
    return filter_image, final_points

def detectLines(points, tolerance):
    points.sort(key=lambda x: x[1])
    lines = []
    for i in range(0, len(points) - 3):
        temp = points[i][0]*points[i+1][1] + points[i+1][0]*points[i+2][1] + points[i+2][0]*points[i][1] - points[i+1][0]*points[i][1] - points[i+2][0]*points[i+1][1] - points[i][0]*points[i+2][1]
        if math.fabs(temp) < tolerance:
            lines.append((points[i][0],points[i][1],points[i+2][0],points[i+2][1],1))
        else:
            for j in range(0, len(lines) - 1):
                if lines[j][0] == points[i][0] and lines[j][1] == points[i][1]:
                    lines[j][2] = points[i+2][0]
                    lines[j][3] = points[i+2][1]
                    lines[j][4] = lines[j][4] + 1
    
    return lines

def trendLine(points):
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y,rcond=None)[0]
    return m,c

def warpingFrame(img):
    WarpingPoints = np.float32([[0, 200], [WIDTH, 200],
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
    HH = np.zeros((100,img.shape[1]), np.uint8)
    if flag == 1:
        for c in range(img.shape[1]):
            cv.line(HH, (c, 100), (c, 100-int(img_x_sum[c]*100/255)),255)
        cv.imshow('HH', HH)
    """ HV = np.zeros((img.shape[0],100), np.uint8)
    for l in range(img.shape[0]):
        cv.line(HV, (0, l),(int(img_y_sum[l]*100/255), l), 255)
    cv.imshow('HV', HV) """
    return img_x_sum, img_y_sum

def rotateImage(img, angle):
    rotated = imutils.rotate_bound(img, angle)
    return rotated

angle_prev = 0
angle_var = 10
angle_step = 1
forward_line_th = 20

n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])
u_array =  np.array([0,0,0,0,0], dtype=float)

def detectLines(frame):
    global angle_prev
    global angle_var
    global angle_step
    global forward_line_th
    global n_array
    global u_array
    
    imgA = ImgProcess(frame)
    max_x = max_y = angle_x = angle_y = 0
    for angle in range( angle_prev + angle_var, angle_prev - angle_var, -angle_step):
        imgB = rotateImage(imgA, angle)
        x_hist, y_hist = drawHist(imgB, 0)
        if x_hist.mean() < 100:
            if x_hist.max() > max_x:
                max_x = x_hist.max()
                angle_x = angle

    if math.fabs(angle_x) > 100:
        if(angle_x > 0):
            angle_x = 100 
        else:
            angle_x = -100
    
    angle_prev = angle_x

    if angle_x > 0:
        angle_x = angle_x - forward_line_th
    elif angle_x < 0:
        angle_x = angle_x + forward_line_th

    """ u_array = np.roll(u_array, -1, axis=0)
    u_array[4] = angle_x
    u_array = np.multiply(u_array, n_array)
    u_mean = np.sum(u_array) """

    return angle_x

########################################
##              MAIN                  ##
########################################

""" s """


""" cap = cv.VideoCapture("Projeto/video4.mp4")
ret, frame = cap.read()

while(ret and cv.waitKey(2) != ord('q')):
    imgA = ImgProcess(frame)

    imgB = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
    imgB2 = imgA.copy()

    drawVirtualCircles(imgB, (WIDTH//2,HEIGHT//3), 70, 450)
    drawVirtualCircles(imgB2, (WIDTH//2,HEIGHT//3), 70, 450)

    imgC = cv.bitwise_and(imgA, imgB)
    imgC, points = pointSelectionFilter(imgC, 50)

    lines = detectLines(points, 1000)

    print(lines)
    if(len(lines) > 0):
        for i in range(0, len(lines) - 1):
            cv.line(imgC, (lines[i][1], lines[i][0]), (lines[i][3], lines[i][2]), 255, 2)
    imgStack = stackImages(0.4,([frame, imgA], [imgB2, imgC]))
    cv.imshow("Images",imgStack) 
    ret, frame = cap.read()
    
cap.release()
cv.destroyAllWindows() """