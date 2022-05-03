from cmath import isnan, pi
from ctypes.wintypes import HWINSTA
import cv2 as cv
from cv2 import determinant
from cv2 import drawContours
import numpy as np
import math
import imutils
from utils import *

WIDTH = 640
HEIGHT = 480
CENTER_COORD = (WIDTH//2,HEIGHT-1)
font = cv.FONT_HERSHEY_COMPLEX

def ImgProcess(img):
    img = cv.resize(img, (WIDTH,HEIGHT))
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    img = cv.blur(img,(5,5))
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)
    img = cv.morphologyEx(img,cv.MORPH_BLACKHAT,(5,5))
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (30,1))
    horizontal_lines = cv.morphologyEx(img, cv.MORPH_CLOSE, horizontal_kernel, iterations=1)
    img = img - horizontal_lines
    return img

def drawVirtualLines(img, center, start_angle, end_angle):
    for i in range (start_angle,end_angle+1,10):
        CalcXFinal = center[0] + (WIDTH//1.5) * math.cos(math.radians(i))
        CalcYFinal = center[1] - (WIDTH//1.5) * math.sin(math.radians(i))
        cv.line(img,(int(center[0]),int(center[1])),(int(CalcXFinal),int(CalcYFinal)),color = 255, thickness= 1)

def drawVirtualCircles(img, center, start_radius, end_radius):
    for i in range (start_radius,end_radius+1,10):
        cv.ellipse(img,center,(i,i),0,15,165,color = 255, thickness= 2)
    return img

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

def trendLine(points):
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y,rcond=None)[0]
    return m,c

def drawTrendLine(slope, b, img,color):
    for i in range(0,WIDTH):
        y = slope * i + b
        cv.line(img,(i,int(y)),(i,int(y)),color,1)

def rotateImage(img, angle):
    rotated = imutils.rotate_bound(img, angle)
    return rotated

def IIRFilter(a,y,x):
    y = a * y + (1 - a) * x
    return round(y)

def MeanFilter(u):
    global n_array
    global u_array 
    u_array = np.roll(u_array, -1, axis=0)
    u_array[4] = u
    u_array = np.multiply(u_array, n_array)
    u_mean = np.sum(u_array)
    return u_mean


PRIORITY_AHEAD = 0
PRIORITY_RIGHT = 1
PRIORITY_LEFT = 2

n_lines = 0
angle = 0
m = 0
c = 0
n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])
u_array =  np.array([0,0,0,0,0], dtype=float)
Kp = 0.20
u = 0
contours = []

def controlProcess(u):
    u = u * Kp
    if (u > 60):
        u = 60
    elif (u < -60):
        u = -60
    u = u * -1      # por causa do simulador
    return u 

def detectLines(frame):
    global n_lines
    imgA = ImgProcess(frame)
    imgB = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
    imgB2 = imgA.copy()

    drawVirtualCircles(imgB, (WIDTH//2,int(HEIGHT//2.5)), 70, 450)
    drawVirtualCircles(imgB2, (WIDTH//2,int(HEIGHT//2.5)), 70, 450)

    imgC = cv.bitwise_and(imgA, imgB)

    lines = cv.HoughLinesP(imgC,cv.HOUGH_PROBABILISTIC, np.pi/180, 10,None, 30,200)
    if lines is not None:
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                cv.polylines(imgC, [pts], True, 255,thickness=25)

    imgC = cv.morphologyEx(imgC, cv.MORPH_ERODE, np.ones((17,17),np.uint8))
    contours, _ = cv.findContours(imgC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    n_lines = IIRFilter(0.4, n_lines, len(contours))
    # print(n_lines)
    return n_lines

def trendLineAlgorithm(frame, priority, n_lines):
    global angle 
    global m
    global c
    global u
    line_points = []
    if(n_lines > 0 and n_lines < 3):
        for contour in contours:
            line_point = cv.approxPolyDP(contour, 3, False)
            line_point = np.squeeze(line_point)
            line_point = list(line_point)
            line_point.sort(key=lambda x: x[0])
            if(len(line_point) > 2):
                m,c = trendLine([[line_point[0][0],line_point[0][1]],[line_point[-1][0],line_point[-1][1]]])
                #drawTrendLine(m, c, imgC, 255)
                line_points.append([line_point[0][0],line_point[0][1],line_point[-1][0],line_point[-1][1], m, c])
                #cv.line(imgC, (0,100),(WIDTH,100), 255, 5)
                #print(line_point[0], line_point[-1])
                #cv.line(imgC, (line_points[0][0], line_points[0][1]), (line_points[-1][0], line_points[-1][1]), 127, 15)
        if (len(line_points) == 0):
            return u
        if(len(line_points) == 1):
            u = (95 - line_points[0][5])/line_points[0][4] - WIDTH//2
        else:
            if (priority == PRIORITY_LEFT or priority == PRIORITY_AHEAD):
                u = (95 - line_points[0][5])/line_points[0][4] - WIDTH//2
            else:
                u = (95 - line_points[1][5])/line_points[1][4] - WIDTH//2
            #u = (u + u2)/2
        u = MeanFilter(u)
        u  = controlProcess(u)
        #print(u)
        return u
    return 0