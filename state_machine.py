from re import X
import numpy as np
import cv2 as cv
from test import WIDTH
from variance_algorithm import twoLineAlgorithm
from warpingHist_algorithm import oneLineAlgorithm
from warpingHist_algorithm import getHistInfo
from warpingHist_algorithm import distinguishLine

NO_LINES = 0
ONE_LINE = 1
TWO_LINES = 2

x = []
y = []
state = NO_LINES
old = NO_LINES  
th = 100 

def encode_FSM(frame):
    global x 
    global y
    global state
    global old
    x, y = getHistInfo(frame)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    u = 0
    frame2 = frame.copy()

    old = state

    if y_mean == 0 and x_mean == 0:                # No lines
        print("No lines detected")
        state = NO_LINES
    if y_mean < 10:                                # One Line 
        ret = distinguishLine(frame)
        if ret == 0:
            print("Left line detected")
        elif ret == 1:
            print("Right line detected")
        else:
            print("Horizontal line detected")
        u = oneLineAlgorithm(frame)
        state = ONE_LINE
    else:                                          # Two Lines
        #print("Two lines detected")
        if (old != TWO_LINES):                     # Restart da média pesada (para não fazer média com valores anteriores)
            u = twoLineAlgorithm(frame2, 1)
        else:
            u = twoLineAlgorithm(frame2, 0)
        state = TWO_LINES

    #cv.imshow('Frame', frame)
    return u