import numpy as np
import cv2 as cv
from trendLine_algorithm import trendLineAlgorithm
from trendLine_algorithm import detectLines
from signals import detectSignal    
from obstacleDetection import detectObstacle
from crosswalk_algorithm import detectCrossWalk, crossWalkControl
from parking import processPark
import time

WIDTH = 640
HEIGHT = 480

# States
STATE_STOP = 0
STATE_CROSSWALK_1 = 1
STATE_RUN = 2
STATE_JUNCTION_1 = 3
STATE_JUNCTION_2 = 4
STATE_CROSSWALK_2 = 5
STATE_OBSTACLE = 6
STATE_PARK = 7
STATE_IDLE = 8
STATE_JUNCTION_2 = 9

# Number of lines
NO_LINES = 0
ONE_LINE = 1
TWO_LINES = 2

# Priority
PRIORITY_AHEAD = 0
PRIORITY_RIGHT = 1
PRIORITY_LEFT = 2

# Car side
SIDE_LEFT = 0
SIDE_RIGHT = 1

# Parking 2 status
NO_OBSTACLE = 0
OBSTACLE_LEFT = 1
OBSTACLE_RIGHT = 2
OBSTACLE_LEFT_AND_RIGHT = 3

# Variables
state = STATE_RUN                                                               # Actual state
next_state = STATE_RUN                                                          # Next state
subState = 0                                                                    # Substate
priority = PRIORITY_LEFT                                                        # Priority
side = SIDE_RIGHT                                                               # Car side
vel = 0                                                                         # Velocity
angle = 0                                                                       # Angle
prev_cross = STATE_CROSSWALK_2                                                  # Previous crosswalk detected
cross_flag = 1                                                                  # Flag to detect crosswalk
angle_save = 0                                                                  # Angle save
vtimer = 0                                                                      # Virtual timer variable

# Stop state
def stop():
    global vel
    global angle
    global next_state
    vel = 0                                                                     # Stop the car
    angle = 0                                                                   # Set the angle to zero
    if (prev_cross == STATE_CROSSWALK_1):                                       # If the previous state was crosswalk 1
        next_state = STATE_CROSSWALK_1                                          # Keep in this state
    elif (prev_cross == STATE_CROSSWALK_2):                                     # If the previous state was crosswalk 2
        next_state = STATE_CROSSWALK_2                                          # Keep in this state
    else:
        print("ERROR: Invalid state")
        next_state = STATE_IDLE

# Crosswalk 1
def crosswalk_1(frame):
    global next_state
    global prev_cross
    global priority
    prev_cross = STATE_CROSSWALK_1                                              # Save the previous state
    ret = detectSignal(frame)                                                   # Detect the signal
    if (ret == "stop"):                                                         # If the signal is stop
        next_state = STATE_STOP                                                 # Change the state
        priority = PRIORITY_LEFT                                                # Set the priority
        print("Stop")
    elif (ret == "finish"):                                                     # If the signal is finish
        next_state = STATE_STOP                                                 # Change the state
        priority = PRIORITY_LEFT                                                # Set the priority
        print("Finish")
    elif (ret == "right"):                                                      # If the signal is right
        priority = PRIORITY_RIGHT                                               # Set the priority
        next_state = STATE_RUN                                                  # Change the state
        print("Right")
    elif (ret == "left"):                                                       # If the signal is left
        priority = PRIORITY_RIGHT                                               # Set the priority   
        next_state = STATE_RUN                                                  # Change the state    
        print("Left")
    elif (ret == "ahead"):                                                      # If the signal is ahead
        priority = PRIORITY_LEFT                                                # Set the priority
        next_state = STATE_RUN                                                  # Change the state
        print("Ahead")
    elif (ret == "park"):                                                       # If the signal is park
        priority = PRIORITY_RIGHT                                               # Set the priority
        next_state = STATE_PARK                                                 # Change the state
        print("Park")
    else:                                                                       # If the signal is none (no signal)
        priority = PRIORITY_LEFT                                                # Set the priority
        next_state = STATE_RUN                                                  # Change the state

# Run state
def run(frame):
    global vel
    global angle
    global next_state
    global cross_flag
    global start
    vel = 3                                                                     # Set the velocity to 3
    if (detectObstacle(frame, angle) == 1):                                     # If there is an obstacle
        next_state = STATE_OBSTACLE                                             # Change the state
        vel = 0                                                                 # Stop the car
        angle = 0                                                               # Set the angle to zero
        return                                                                  # Return
    _,n_lines,_ = detectLines(frame, 0)                                             # Detect the lines 
    isCrossWalk = detectCrossWalk(frame)                                        # Detect the crosswalk
    if (n_lines == NO_LINES and prev_cross == STATE_CROSSWALK_1 and\
        isCrossWalk == 0):                                                      # If the car is in the junction
        next_state = STATE_JUNCTION_1                                           # Change the state
        vel = 2                                                                 # Set the velocity to 2
        cross_flag = 0                                                          # Reset the crosswalk flag
    elif (isCrossWalk and prev_cross == STATE_CROSSWALK_1 and cross_flag == 0): # If the car is in the crosswalk 2
        next_state = STATE_CROSSWALK_2                                          # Change the state
        vel = 1                                                                 # Set the velocity to 1
        cross_flag = 1                                                          # Set the crosswalk flag
    elif (isCrossWalk and prev_cross == STATE_CROSSWALK_2 and cross_flag == 0): # If the car is in the crosswalk 1
        next_state = STATE_CROSSWALK_1                                          # Change the state
        vel = 1                                                                 # Set the velocity to 1
        cross_flag = 1                                                          # Set the crosswalk flag
    elif (isCrossWalk == 0):                                                    # If the car is not in the crosswalk
        next_state = STATE_RUN                                                  # The state is run
        cross_flag = 0                                                          # Reset the crosswalk flag
    else:                                                                       # Else
        next_state = STATE_RUN                                                  # The state is run

    if (isCrossWalk == 1):                                                      # If the car is in the crosswalk
        vel = 1                                                                 # Set the velocity to 1
        angle = crossWalkControl(frame)                                         # Set the angle
    else:                                                                       # If the car is not in the crosswalk
        angle = trendLineAlgorithm(frame, priority, n_lines)                    # Set the angle

# Junction 1 state
def junction_1(frame):
    global vel
    global angle
    global next_state
    _, n_lines, _ = detectLines(frame, 1)                                             # Detect the lines
    if (n_lines == TWO_LINES):                                                  # If the car is in the junction
        next_state = STATE_RUN                                                  # Change the state
        angle = trendLineAlgorithm(frame, priority, n_lines)                    # Set the angle
    else:                                                                       # Else
        next_state = STATE_JUNCTION_1                                           # Change the state
        angle = 5                                                               # Set the angle

# Crosswalk 2 state
def crosswalk_2(frame): 
    global next_state
    global prev_cross
    global priority
    prev_cross = STATE_CROSSWALK_2                                              # Save the previous state
    ret = detectSignal(frame)                                                   # Detect the signal
    if (ret == "stop"):                                                         # If the signal is stop
        next_state = STATE_STOP                                                 # Change the state
        priority = PRIORITY_RIGHT                                               # Set the priority
        print("Stop")   
    elif (ret == "finish"):                                                     # If the signal is finish
        next_state = STATE_STOP                                                 # Change the state
        priority = PRIORITY_RIGHT                                               # Set the priority
        print("Finish")
    elif (ret == "right"):                                                      # If the signal is right
        priority = PRIORITY_RIGHT                                               # Set the priority
        next_state = STATE_RUN                                                  # Change the state
        print("Right")
    elif (ret == "left"):                                                       # If the signal is left
        priority = PRIORITY_RIGHT                                               # Set the priority
        next_state = STATE_JUNCTION_2                                           # Change the state
        print("Left")
    elif (ret == "ahead"):                                                      # If the signal is ahead
        priority = PRIORITY_RIGHT                                               # Set the priority
        next_state = STATE_RUN                                                  # Change the state
        print("Ahead")
    elif (ret == "park"):                                                       # If the signal is park
        priority = PRIORITY_RIGHT                                               # Set the priority 
        next_state = STATE_PARK                                                 # Change the state
        print("Park")
    else:                                                                       # If the signal is none (no signal)
        priority = PRIORITY_RIGHT                                               # Set the priority
        next_state = STATE_RUN                                                  # Change the state

# Obstacle state
def obstacle(frame):
    global vel
    global angle
    global next_state
    global subState
    global angle_save
    global vtimer
    global priority
    next_state = STATE_OBSTACLE                                                 # Change the state
    vel = 2                                                                     # Set the velocity to 2.5
    if (subState == 0):                                                         # Detected the obstacle  
        if (side == SIDE_RIGHT):                                                # If the car is on the right side of the road
            priority = PRIORITY_RIGHT                                           # Set the priority
            angle = 30                                                          # Turn left in order to avoid the obstacle
        else:                                                                   # If the car is on the left side of the road
            angle = -30                                                         # Turn right in order to avoid the obstacle
        angle_save = angle                                                      # Save the angle
        subState = 1                                                            # Update the sub-state to 1
        vtimer = time.time()                                                    # Reset the time

    elif (subState == 1):                                                       # Avoid the obstacle
        _, n_lines, _ = detectLines(frame, 0)                                         # Detect the lines
        if (time.time()-vtimer >= 2):                                           # If there are two lines and the time is greater than 1.5 seconds
            subState = 2                                                        # The car already passed the obstacle
            angle = trendLineAlgorithm(frame, priority, n_lines)                # Calculate the angle
            vtimer = time.time()                                                # Reset the time
        else:                                                                   # If the car is still in the obstacle
            angle = angle_save                                                  # Keep the same angle
            subState = 1                                                        # The car is still in the obstacle
    
    elif (subState == 2):                                                       # Walk in this lane for a while
        if (time.time()-vtimer >= 5):                                           # If the time is greater than 3 seconds
            angle = -1 * angle_save                                             # Turn the car to the oposite angle saved
            subState = 3                                                        # Update the sub-state to 3
            vtimer = time.time()                                                # Reset the time
        else:                                                                   # If the time is less than x seconds
            _, n_lines, _ = detectLines(frame, 0)                                     # Detect the lines
            angle = trendLineAlgorithm(frame, priority, n_lines)                # Calculate the angle
            subState = 2                                                        # The car is still in the sub-state 2
            vel = 3                                                             # Set the velocity to 3

    elif (subState == 3):                                                       # The car must turn back to the lane 
        _, n_lines, _ = detectLines(frame, 0)                                         # Detect the lines
        if (time.time()-vtimer >= 1.5):                                           # If there are two lines
            next_state = STATE_RUN                                              # Go to the RUN state, because the car is in the lane
            vel = 3                                                             # Set the velocity to 3
            subState = 0                                                        # Reset the sub state for new iterations
            angle = trendLineAlgorithm(frame, priority, n_lines)                # Calculate the angle
            vtimer = time.time()                                                # Reset the time
            if (side == SIDE_RIGHT):                                            # If the car is on the right side of the road
                priority = PRIORITY_LEFT                                        # Set the priority
        else:                                                                   # If there are not two lines
            vel = 2                                                             # Set the velocity to 2.5
            angle = -1 * angle_save                                             # Keep turning the car to the oposite angle saved
            subState = 3                                                        # The car is still in the sub-state 3
    else:                                                                       # If the sub-state is not 0, 1, 2 or 3
        print("Obstacle ERROR")                                     
        vel = 0                                                                 # Set the velocity to 0
        angle = 0                                                               # Set the angle to 0
        vtimer = time.time()                                                    # Reset the time

""" 
# Park state
def park(img):
    global vel
    global angle
    global next_state
    global subState
    global angle_save
    global vtimer
    next_state = STATE_PARK                                                     # Change the state
    vel = 2                                                                     # Set the velocity to 2
    if (detectObstacle(img, angle) == 1):                                       # If there is an obstacle
        print("Impossible to park: Obstacle detected")                          # Print the error message
        next_state = STATE_IDLE                                                 # Change the state
        vel = 0                                                                 # Stop the car
        angle = 0                                                               # Set the angle to zero
        return                                                                  # Return
    if (subState == 0 and detectCrossWalk(img) == 0):                           # The car detected the park signal
        angle = -15                                                             # Turn left
        angle_save = angle                                                      # Save the angle
        subState = 1                                                            # Update the sub-state to 1
        vtimer = time.time()                                                    # Reset the time
    elif (subState == 0):
        angle = crossWalkControl(img)                                           # Calculate the angle
    elif (subState == 1):                                                       # The car is turning left, in order to move to the parking area
        n_lines = detectLines(img, 0)                                           # Detect the lines
        if (time.time()-vtimer >= 3.5):                                         # If the car is in the parking area
            subState = 2                                                        # Update the sub-state to 2
            angle = trendLineAlgorithm(img, priority, n_lines)                  # Calculate the angle   
            vtimer = time.time()                                                # Reset the time
        else:                                                                   # If the car is still turning left
            angle = angle_save                                                  # Keep turning the car to the oposite angle saved
            subState = 1                                                        # The car is still in the sub-state 1
    elif (subState == 2):                                                       # The car is in the parking area but will correct the angle
        n_lines = detectLines(img, 0)                                           # Detect the lines
        angle = trendLineAlgorithm(img, priority, n_lines)                      # Calculate the angle
        if (time.time()-vtimer >= 4):                                           # If the car is straight    
            subState = 0                                                        # Reset the sub-state for new iterations
            next_state = STATE_IDLE                                             # Go to the IDLE state (Park is finished)
            vtimer = time.time()                                                # Reset the time
        else:                                                                   # If the car is not straight
            subState = 2                                                        # The car is still in the sub-state 2
    else:                                                                       # If the car is in an error state
        print("Park ERROR")
        vel = 0                                                                 # Set the velocity to 0
        angle = 0                                                               # Set the angle to 0
        vtimer = time.time()                                                    # Reset the time 
 
""" 
def park(img):
    global vel
    global angle
    global next_state
    global subState
    global angle_save
    global vtimer
    global parkStatus
    next_state = STATE_PARK                                             # Change the state
    vel = 2                                                             # Set the velocity to 2
    if (subState == 0):                                                 # The car detected the park signal
        vel = -2                                                        # Set the velocity to -2 (backward)
        angle = 40                                                      # Turn backward left
        angle_save = angle                                              # Save the angle
        subState = 1                                                    # Update the sub-state to 1
        vtimer = time.time()                                            # Reset the time
    elif (subState == 1):                                               # The car is turning backward left, in order to position to the parking area
        if (time.time()-vtimer >= 3.4):                                 # If the car already made the maneuver
            subState = 2                                                # Update the sub-state to 2 
            vtimer = time.time()                                        # Reset the time
        else:                                                           # If the car is still turning left
            angle = angle_save                                          # Keep turning the car to the angle saved
            vel = -2                                                    # Set the velocity to -2 (backward)
            subState = 1                                                # The car is still in the sub-state 1
    elif (subState == 2):                                               # The car is in the parking area but will correct the angle
        if (time.time()-vtimer >= 5):                                   # If the car is in the parking area
            vel = 0                                                     # Set the velocity to 0
            angle = 0                                                   # Set the angle to 0
            subState = 3                                                # Update the sub-state to 3
            vtimer = time.time()                                        # Reset the time
        else:                                                           # If the car is still turning left
            angle, parkStatus = processPark(img)                        # Process Park                                                   # Keep moving straight
            if (parkStatus == OBSTACLE_LEFT_AND_RIGHT):                 # If it is impossible to park
                next_state = STATE_IDLE                                 # Go to the IDLE state 
            else:                                                       # If it is possible to park
                vel = 1.5                                               # Set the velocity to -1.5 (backward)
                subState = 2                                            # The car is still in the sub-state 2
    elif (subState == 3):                                               # The car is in the parking area but will correct the angle
        if (time.time()-vtimer >= 1.75):                                # If the car is in the parking area
            vel = 0                                                     # Set the velocity to 0
            angle = 0                                                   # Set the angle to 0
            subState = 0                                                # Reset the sub-state for new iterations
            vtimer = time.time()                                        # Reset the time
            next_state = STATE_IDLE                                     # Go to the IDLE state (Park is finished)
        else:                                                           # If the car is still turning left
            vel = 1
            if (parkStatus == OBSTACLE_RIGHT):
                angle = -20
            else:
                angle = 25
            subState = 3                                                # The car is still in the sub-state 3
    else:                                                               # If the car is in an error state
        print("Park ERROR")
        vel = 0                                                         # Set the velocity to 0
        angle = 0                                                       # Set the angle to 0
        vtimer = time.time()                                            # Reset the time 

# Junction 2 state
def junction_2(img):
    global vel
    global angle
    global next_state
    global subState
    global vtimer
    next_state = STATE_JUNCTION_2                                       # Change the state
    vel = 3                                                             # Set the velocity to 3
    if (detectObstacle(img, angle) == 1):                               # If there is an obstacle
        print("Impossible to go to the junction: Obstacle detected")    # Print the error message
        next_state = STATE_IDLE                                         # Change the state
        vel = 0                                                         # Stop the car
        angle = 0                                                       # Set the angle to zero
        return                                                          # Return
    if (subState == 0 and detectCrossWalk(img) == 0):                   # The car detected the left signal and is going to the junction
        _, n_lines, _ = detectLines(img, 0)                                   # Detect the lines
        angle = trendLineAlgorithm(img, priority, n_lines)              # Calculate the angle
        subState = 1                                                    # Update the sub-state to 1 
        vtimer = time.time()                                            # Reset the time
    elif (subState == 0):                                               # The car still going in crosswalk
        vel = 1                                                         # Set the velocity to 1
        angle = crossWalkControl(img)                                   # Calculate the angle
        subState = 0                                                    # The car is still in the sub-state 0   
    elif (subState == 1):                                        
        if (time.time()-vtimer >= 3):                                   # If the car is in the junction
            angle = 15                                                  # Turn left
            subState = 2                                                # Update the sub-state to 2
            vtimer = time.time()                                        # Reset the time
        else:                                                           # If the car is not in the junction
            subState = 1                                                # The car is still in the sub-state 1
            _, n_lines, _ = detectLines(img, 0)                               # Detect the lines
            angle = trendLineAlgorithm(img, priority, n_lines)          # Calculate the angle
    elif (subState == 2):                                               # The car is in the junction
        if (time.time()-vtimer >= 3):                                   # If the car is in the junction
            _, n_lines, _ = detectLines(img, 0)                               # Detect the lines
            angle = trendLineAlgorithm(img, priority, n_lines)          # Calculate the angle
            subState = 0                                                # Reset the sub-state for new iterations
            next_state = STATE_RUN                                      # Go to the IDLE state (Junction is finished)
            vtimer = time.time()                                        # Reset the time
        else:                                                           # If the car is not in the junction
            angle = 15                                                  # keep turning left
            subState = 2                                                # The car is still in the sub-state 2
    else:                                                               # If the car is in an error state
        print("Junction Left ERROR")
        vel = 0                                                         # Set the velocity to 0
        angle = 0                                                       # Set the angle to 0
        vtimer = time.time()                                            # Reset the time 

# Idle state
def idle():
    global vel
    global angle
    global next_state
    print("Idle State")
    vel = 0                                                             # Set the velocity to 0
    angle = 0                                                           # Set the angle to 0
    next_state = STATE_IDLE                                             # Keep in this state

# Encode FSM (Finite State Machine)
def encode_FSM(frame, signalFrame):
    global state
    global next_state
    global priority
    global side
    global prev_cross
    global vel
    global angle

    if (state == STATE_STOP):                                       
        #print("Stop")
        stop()                                                          
    elif (state == STATE_RUN):
        #print("Run")
        run(frame)
    elif (state == STATE_CROSSWALK_1):
        print("Crosswalk 1")
        crosswalk_1(signalFrame)
    elif (state == STATE_JUNCTION_1):
        #print("Junction 1")
        junction_1(frame)
    elif (state == STATE_CROSSWALK_2):
        print("Crosswalk 2")
        crosswalk_2(signalFrame)
    elif (state == STATE_OBSTACLE):
        #print("Obstacle")
        obstacle(frame)
    elif (state == STATE_PARK):
        #print("Park")
        park(frame)
    elif (state == STATE_JUNCTION_2):
        #print("Junction 2")
        junction_2(frame)
    else:
        idle()

    state = next_state
    frame = cv.resize(frame, (WIDTH, HEIGHT))
    cv.imshow("Frame", frame)
    return vel, angle