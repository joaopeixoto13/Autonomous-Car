from cmath import isnan, pi
import cv2 as cv
from cv2 import determinant
import numpy as np
import math
import imutils

WIDTH = 640
HEIGHT = 480

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