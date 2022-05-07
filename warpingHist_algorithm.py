from cmath import isnan, pi
import cv2 as cv
from cv2 import determinant
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
    img = warpingFrame(img)
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    img = cv.blur(img,(5,5))
    _,img = cv.threshold(img,170,255,cv.THRESH_BINARY)
    return img

def distinguishLine(frame):
    # 0 if Left, 1 if Right, 2 if Horizontal
    center = WIDTH//2
    th_factor = 0.125
    th_sum = 1.5
    img = ImgProcess(frame)
    img_left = img[:,:center-int(center*th_factor)]
    img_right = img[:,center+int(center*th_factor):]
    x_left = np.sum(img_left, axis=0)
    x_right = np.sum(img_right, axis=0)
    x_left = x_left/255
    x_right = x_right/255
    x_left = np.sum(x_left)
    x_right = np.sum(x_right)

    print(x_left, x_right)

    if ((x_left == 0 and x_right== 0) or (int(max(x_left, x_right)-min(x_left, x_right)) <= 200)):
        return 2
    elif (x_left > int(th_sum*x_right)):
        return 0
    elif (x_right > int(th_sum*x_left)):
        return 1
    return 3

def rotateImage(img, angle):
    rotated = imutils.rotate_bound(img, angle)
    return rotated

def getHistInfo(frame):
    img = ImgProcess(frame)
    x, y = drawHist(img, 0)
    return x, y

def controlProcess(u):
    if (u > 60):
        u = 60
    elif (u < -60):
        u = -60 
    u = Kp * u
    #u = u * -1
    return u

def warpingAlgorithm(imgA):
    global angle_prev
    global angle_var
    global angle_step
    global forward_line_th
    global angle_x
    max_x = 0
    imgA = ImgProcess(imgA)
    for angle in range(angle_prev + angle_var, angle_prev - angle_var, -angle_step):
        imgB = rotateImage(imgA, angle)
        x_hist, y_hist = drawHist(imgB, 0)
        if x_hist.mean() < 100:
            diff = np.diff(x_hist)
            if diff.max() > max_x:
                max_x = diff.max()
                angle_x = angle
        else:
            print("Passadeira")
    #cv.imshow('warp', rotateImage(imgA, angle_x))

    if math.fabs(angle_x) > 100:
        if(angle_x > 0):
            angle_x = 100 
        else:
            angle_x = -100
    
    angle_prev = int(angle_x)

    if angle_x > 0:
        angle_x = angle_x - forward_line_th
    elif angle_x < 0:
        angle_x = angle_x + forward_line_th

    print(f"Angle_x: {angle_x}")
    angle_x = controlProcess(angle_x)
    return angle_x

angle_prev = 0
angle_var = 36
angle_step = 3
forward_line_th = 10
angle_x = 0
Kp = 0.65 