import math
import cv2 as cv
import numpy as np

WIDTH = 640
HEIGHT = 480
BLOCK_NUMBER = 25000
BLOCK_X_TH = WIDTH//2
BLOCK_Y_TH = HEIGHT//2.5
BLOCK_X_TH_TH = 80

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


def filterBlock(img):
    lower = np.array([0, 150, 0], dtype = "uint8")
    upper = np.array([100, 255, 100], dtype = "uint8")
    mask = cv.inRange(img, lower, upper)
    mask = cv.bitwise_and(img, img, mask=mask)
    return mask

block = cv.imread("Projeto/data/block2.png")
block = cv.resize(block, (WIDTH,HEIGHT))
cv.imshow("block1", block)
block = filterBlock(block)
gray_block = cv.cvtColor(block, cv.COLOR_BGR2GRAY)
_, gray_block = cv.threshold(gray_block, 50, 255, cv.THRESH_BINARY)
pos = np.nonzero(gray_block)
print(len(pos[0]))
print(pos)

if len(pos[0]) > BLOCK_NUMBER and pos[0][-1] >= BLOCK_Y_TH and math.fabs(pos[1][(len(pos[1])-1) //2] - BLOCK_X_TH) < BLOCK_X_TH_TH:
    print("Block detected")

cv.imshow("block", gray_block)
cv.waitKey(0)
