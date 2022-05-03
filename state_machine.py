import numpy as np
import cv2 as cv
from test import WIDTH
#from variance_algorithm import twoLineAlgorithm
#from warpingHist_algorithm import oneLineAlgorithm
#from warpingHist_algorithm import getHistInfo
#from warpingHist_algorithm import distinguishLine
from trendLine_algorithm import trendLineAlgorithm
from trendLine_algorithm import detectLines
from signals import detectSignal    
from obstacleDetection import detectObstacle

STATE_STOP = 0
STATE_CROSSWALK_1 = 1
STATE_RUN = 2
STATE_JUNCTION_1 = 3
STATE_CROSSWALK_2 = 4
STATE_OBSTACLE = 5
STATE_IDLE = 6

NO_LINES = 0
ONE_LINE = 1
TWO_LINES = 2

PRIORITY_AHEAD = 0
PRIORITY_RIGHT = 1
PRIORITY_LEFT = 2

SIDE_LEFT = 0
SIDE_RIGHT = 1

prev_cross = STATE_CROSSWALK_1 
state = STATE_STOP
next_state = STATE_STOP
priority = PRIORITY_LEFT
side = SIDE_RIGHT
vel = 0
angle = 0

def stop():
    vel = 0
    angle = 0
    if (prev_cross == STATE_CROSSWALK_1):
        next_state = STATE_CROSSWALK_1
    elif (prev_cross == STATE_CROSSWALK_2):
        next_state = STATE_CROSSWALK_2
    else:
        print("ERROR: Invalid state")
        next_state = STATE_IDLE
    return next_state, vel , angle

def crosswalk_1(frame):
    prev_cross = STATE_CROSSWALK_1
    ret = detectSignal(frame)
    if (ret == "stop"):
        next_state = STATE_STOP
        print("Stop")
    elif (ret == "finish"):
        next_state = STATE_STOP 
        print("Finish")
    elif (ret == "right"):
        priority = PRIORITY_RIGHT
        next_state = STATE_RUN
        print("Right")
    elif (ret == "left"):
        priority = PRIORITY_LEFT
        next_state = STATE_RUN
        print("Left")
    elif (ret == "ahead"):
        priority = PRIORITY_LEFT
        next_state = STATE_RUN
        print("Ahead")
    elif (ret == "park"):
        print("Park")
    else:
        print("No signal")
    return next_state, prev_cross , priority

def run(frame):
    vel = 2.5
    """ if (detectObstacle(frame) == 1):
        next_state = STATE_OBSTACLE
        return """
    n_lines = detectLines(frame)
    if (n_lines == NO_LINES and prev_cross == STATE_CROSSWALK_1):
        next_state = STATE_JUNCTION_1
    elif (n_lines > TWO_LINES and prev_cross == STATE_CROSSWALK_1):
        next_state = STATE_CROSSWALK_2
    elif (n_lines > TWO_LINES and prev_cross == STATE_CROSSWALK_2):
        next_state = STATE_CROSSWALK_1
    else:
        next_state = STATE_RUN
    angle = trendLineAlgorithm(frame, priority, n_lines)
    return next_state, vel, angle

def junction_1(frame):
    n_lines = detectLines(frame)
    if (n_lines == TWO_LINES):
        next_state = STATE_RUN
    else:
        next_state = STATE_JUNCTION_1
    angle = 0
    return next_state, angle

def crosswalk_2(frame):
    prev_cross = STATE_CROSSWALK_2
    ret = detectSignal(frame)
    if (ret == "stop"):
        next_state = STATE_STOP
        print("Stop")
    elif (ret == "finish"):
        next_state = STATE_STOP 
        print("Finish")
    elif (ret == "right"):
        priority = PRIORITY_RIGHT
        next_state = STATE_RUN
        print("Right")
    elif (ret == "left"):
        priority = PRIORITY_LEFT
        next_state = STATE_RUN
        print("Left")
    elif (ret == "ahead"):
        priority = PRIORITY_LEFT
        next_state = STATE_RUN
        print("Left")
    elif (ret == "park"):
        print("Park")
        priority = PRIORITY_LEFT
        next_state = STATE_RUN
    else:
        print("No signal")
        priority = PRIORITY_LEFT
        next_state = STATE_RUN
    return next_state, prev_cross , priority

def obstacle(frame):
   print("Obstacle")

def idle():
    print("Idle State")
    next_state = STATE_IDLE
    return next_state


def encode_FSM(frame, signalFrame):
    global state
    global next_state
    global priority
    global side
    global prev_cross
    global vel
    global angle
    
    print("Encode FSM")

    if (state == STATE_STOP):
        print("Stop")
        next_state, vel, angle = stop()
    elif (state == STATE_RUN):
        print("Run")
        next_state, vel, angle = run(frame)
    elif (state == STATE_CROSSWALK_1):
        print("Crosswalk 1")
        next_state, prev_cross , priority = crosswalk_1(signalFrame)
    elif (state == STATE_JUNCTION_1):
        print("Junction 1")
        next_state, angle = junction_1(frame)
    elif (state == STATE_CROSSWALK_2):
        print("Crosswalk 2")
        next_state, prev_cross , priority = crosswalk_2(signalFrame)
    elif (state == STATE_OBSTACLE):
        print("Obstacle")
        obstacle(frame)
    else:
        print("Idle")
        next_state = idle()

    state = next_state
    return vel, angle



""" NO_LINES = 0
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
        # u = oneLineAlgorithm(frame)
        if (old != ONE_LINE):
            u = trendLineAlgorithm(frame, 1)
        else:
            u = trendLineAlgorithm(frame, 0)
        state = ONE_LINE
    else:                                          # Two Lines
        #print("Two lines detected")
        if (old != TWO_LINES):                     # Restart da média pesada (para não fazer média com valores anteriores)
            u = trendLineAlgorithm(frame, 1)
            #u = twoLineAlgorithm(frame2, 1)
        else:
            u = trendLineAlgorithm(frame, 0)
            #u = twoLineAlgorithm(frame2, 0)
        state = TWO_LINES

    #cv.imshow('Frame', frame)
    return u  """