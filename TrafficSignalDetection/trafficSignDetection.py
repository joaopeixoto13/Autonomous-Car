from re import X
import cv2 as cv
from cv2 import threshold
from cv2 import drawContours
import numpy as np
import math
import json
WIDTH = HEIGHT = 300                                                                                    # Size of the image

# Relations between areas of shapes and the minimum enclosing circle
RELATION_AREA_OF_CIRCLE_AND_TRIANGLE = 2.4184                                                           # 4*PI/(3*SQRT(3))
RELATION_AREA_OF_CIRCLE_AND_SQUARE = 1.5708                                                             # PI/2
RELATION_AREA_OF_CIRCLE_AND_CIRCLE = 1  

# Color Boundaries for the signal
ColorBoundaries = [
    ([0, 100, 100], [20, 255, 255],"RED"),                                                              # Red (0-20)
    ([160, 100, 100], [180, 255, 255],"RED"),                                                           # Red (160-180)
    ([60, 100, 100], [70, 255, 255],"GREEN"),                                                           # Green
    ([95, 100, 100], [120, 255, 255],"BLUE"),                                                           # Blue
    ([25, 100, 100], [30, 255, 255],"YELLOW"),                                                          # Yellow
    ([0, 0, 200], [230, 50, 255],"WHITE"),                                                              # White
    ([0, 0, 0], [230, 255, 50],"BLACK")                                                                 # Black
]

# Minimum number of pixels for a color to be detected
COLOR_MIN_TH = 500

# DataBase
# https://www.rhinocarhire.com/Drive-Smart-Blog/Drive-Smart-Portugal/Portugal-Road-Signs.aspx
# Load data base

# Rectangle
crossWalkImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/CROSSWALK.jpg")
parkImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/PARK.jpg")
priorityImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/PRIORITY.jpg")
roadBendAheadImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/ROAD_BEND_AHEAD.jpg")
# Circle
speedLimitImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/SPEED_LIMIT.jpg")
entryNotAllowedImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/ENTRY_NOT_ALLOWED.jpg")
mandatoryLeftImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/MANDATORY_LEFT.jpg")
mandatoryLightImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/MANDATORY_LIGHTs.jpg")
parkingForbidenImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/PARKING_FORBIDDEN.jpg")
# Triangle
roadBendRightImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/ROAD_BEND_RIGHT.jpg")
roadWorkImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/ROADWORK.jpg")
steppAheadImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/STEPP_AHEAD.jpg")
traficLightAheadImg = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/TRAFIC_LIGHT_AHEAD.jpg")

# Test images
test1 = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/teste1.png")
test2 = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/teste2.png")
test3 = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/teste3.png")
test4 = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/teste4.png")
test5 = cv.imread("Projeto/TrafficSignalDetection/TrafficSignals/teste5.png")

test1 = cv.resize(test1, (WIDTH, HEIGHT))
test2 = cv.resize(test2, (WIDTH, HEIGHT))
test3 = cv.resize(test3, (WIDTH, HEIGHT))
test4 = cv.resize(test4, (WIDTH, HEIGHT))
test5 = cv.resize(test5, (WIDTH, HEIGHT))

trafficSignals = [crossWalkImg, parkImg, priorityImg, roadBendAheadImg, speedLimitImg, entryNotAllowedImg, mandatoryLeftImg, mandatoryLightImg, parkingForbidenImg, roadBendRightImg, roadWorkImg, steppAheadImg, traficLightAheadImg,test1,test2,test3,test4,test5]




# Detect the shape of the signal and return the image without the background
def detectSignalForm(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                                                                                  # Convert to gray scale
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)                                                                                 # Blur image
    _, img_gray = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)                                                            # Threshold image
    contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)                                                   # Find contours
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    signalContour = contours[0]
                                                                                                                                    
    imgNoBack = np.zeros_like(img)                                                                                                  # Create a black image (Remove background)
    imgNoBack[:,:,0] = 255                                                                                                          # Background color (255,0,255), not used in Traffic Signals
    imgNoBack[:,:,2] = 255                                                                                                          
    cv.drawContours(imgNoBack, [signalContour], -1, (255, 255, 255), thickness=cv.FILLED)                                           # Draw the contour
    imgNoBack = cv.bitwise_and(img, imgNoBack)                                                                                      # Add the original image
    temp = signalContour                                                                                                            # Save the contour
    temp.squeeze()                                                                                                                  # Remove the extra dimension
    x_min = temp[:, 0, 0].min()                                                                                                     # Find the minimum x value
    x_max = temp[:, 0, 0].max()                                                                                                     # Find the maximum x value
    y_min = temp[:, 0, 1].min()                                                                                                     # Find the minimum y value
    y_max = temp[:, 0, 1].max()                                                                                                     # Find the maximum y value
    imgNoBack = imgNoBack[y_min:y_max, x_min:x_max]                                                                                 # Crop the image

    AreaSignal = cv.contourArea(signalContour)                                                                                      # Area of the signal                                                                  
    _,radius = cv.minEnclosingCircle(signalContour)                                                                                 # Radius of the minimum enclosing circle
    AreaCircle = math.pi * radius ** 2                                                                                              # Area of the minimum enclosing circle

    if math.fabs((AreaCircle/ AreaSignal) - RELATION_AREA_OF_CIRCLE_AND_CIRCLE) < RELATION_AREA_OF_CIRCLE_AND_CIRCLE*0.1:           # Check if is a circle(0.1% of error because the image are not a perfect circle)
        return "CIRCLE", imgNoBack
    elif math.fabs((AreaCircle/ AreaSignal) - RELATION_AREA_OF_CIRCLE_AND_SQUARE) < RELATION_AREA_OF_CIRCLE_AND_SQUARE*0.15:        # Check if is a square(0.15% of error because the image are not a perfect square)
        return "SQUARE", imgNoBack
    elif math.fabs((AreaCircle/ AreaSignal) - RELATION_AREA_OF_CIRCLE_AND_TRIANGLE) < RELATION_AREA_OF_CIRCLE_AND_TRIANGLE*0.25:    # Check if is a triangle(0.25% of error because the image are not a perfect triangle)
        return "TRIANGLE", imgNoBack
    else:
        return "UNKNOWN", imgNoBack

# Detect the possible colors of the signal( Only the colors in the database )
def detectColorsofSignal(img): 
    colors = []
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)                                                                                    # Convert to HSV
    img_hsv = cv.GaussianBlur(img_hsv, (5, 5), 0)                                                                                   # Blur image
    for color in ColorBoundaries:
        lower = np.array(color[0], dtype="uint8")
        upper = np.array(color[1], dtype="uint8")
        mask = cv.inRange(img_hsv, lower, upper)                                                                                    # Create a mask
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, np.ones((3,3),np.uint8))                                                       # Erode the mask(Eliminate noise)    
        if cv.countNonZero(mask) > COLOR_MIN_TH:                                                                                    # Check if the mask has at least an amount of pixels
            colors.append(color[2])                                                                                                 # Add the color to the list
    return colors

# Detect the possible colors of the pixel( Only the colors in the database )
def detectPixelColor(pixel):
    for color in ColorBoundaries:                                                                                                  
        lower = np.array(color[0], dtype="uint8")                                                                                   
        upper = np.array(color[1], dtype="uint8")                                                                                   
        if lower[0] <= pixel[0] <= upper[0] and lower[1] <= pixel[1] <= upper[1] and lower[2] <= pixel[2] <= upper[2]:             
            return color[2]
    return "UNKNOWN"    

# Detect the % of colors in each "quadrant" of the signal (Only the colors in the database) (C++ is faster, made in python for educational purposes)                  
def MonteCarloSignalDetection(img):                                           
    nPoints = 5000                                                                                                                  # Number of points to test
    colors = [] 
    count = []
    quadrant = []
    quadrants = []
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)                                                                                    # Convert to HSV      
    xIntervals = [[0,img.shape[1]/2],[img.shape[1]/2,img.shape[1]]]                                                                 # X intervals
    yIntervals = [[0,img.shape[0]/2],[img.shape[0]/2,img.shape[0]]]                                                                 # Y intervals
    for xInterval in xIntervals:                                                                                                    # For each quadrant
        colors.clear()                                                                                                                        
        count.clear()
        for yInterval in yIntervals:    
            XrandCoords = np.random.default_rng().uniform(xInterval[0],xInterval[1], (nPoints,))                                    # Generate random x coordinate
            YrandCoords = np.random.default_rng().uniform(yInterval[0],yInterval[1], (nPoints,))                                    # Generate random y coordinate
            for i in range(nPoints):                                                                                                # For each ramdom point  
                XrandCoord = int(XrandCoords[i])    
                YrandCoord = int(YrandCoords[i])
                color = detectPixelColor(img_hsv[YrandCoord,XrandCoord])                                                            # Detect the color of the pixel
                if color in colors:                                                                                                 # If the color is already in the list
                    count[colors.index(color)] = count[colors.index(color)] + 1                                                     # Add 1 to the count   
                else:                                                                                                               # If the color is not in the list
                    colors.append(color)                                                                                            # Add the color to the list
                    count.append(1)                                                                                                 # Add 1 to the count
            count = [x/nPoints for x in count]                                                                                      # Calculate the % of each color
            quadrant = list(zip(colors,count))                                                                                               
            quadrant.sort(key=lambda x: x[1], reverse=True)                                                                         # Sort the quadrant by % of each color
            quadrants.append(quadrant)                                                                                              # Add the quadrant to the list
    return quadrants    

# Detect the signal based on Database
def detectSignal(img):
    signalName = []                                                                                                                 # List of the names of the signals
    shape,imgNoBack = detectSignalForm(img)                                                                                         # Detect the shape of the signal
    colors = detectColorsofSignal(imgNoBack)                                                                                        # Detect the colors of the signal
    for i in dataBase:                                                                                                              # For each signal in the database
        for signal,data in i.items():
            if shape == data["shape"] and colors == data["colors"]:                                                                 # Check if the signal is the same
                signalName.append(signal)                                                                                           # Add the name of the signal to the list     

    if len(signalName) == 0:                                                                                                        # If the signal is not in the database
        return "UNKNOWN"                                                                                                            # Return "UNKNOWN"
    elif len(signalName) == 1:                                                                                                      # If the signal is in the database
        return signalName[0]                                                                                                        # Return the name of the signal
    # If the signal is in the database but is not unique (Doubtful)
    quadrants = MonteCarloSignalDetection(imgNoBack)                                                                                # Detect the % of colors in each "quadrant" of the signal
    
    counts = []                                                                                                                     # List of the counts of each signal
    for i in range(0,len(signalName)):                                                                                              # For each signal in the database
        counts.append(0)                                                                                                            # Add 0 to the list

    for name in signalName:                                                                                                         # For each signal in the doubtful list
        for i in dataBase:                                                                                                          # For each signal in the database
            for signal,data in i.items():                                                                                           # Unpack the data of the signal
                if name == signal:                                                                                                  # If the signal is the same
                    for key,value in data["quadrants"].items():                                                                     # Extract the quadrants of the signal                                                                                 
                            for color,probability in quadrants[int(key) - 1]:                                                       # For each quadrant in signal
                                temp = np.argwhere(np.array(value) == color)                                                        # Find the index of the color in the signal
                                if len(temp) != 0 and math.fabs(value[temp[0][1]][1] - probability) < value[temp[0][1]][1]*0.05:    # If the % of the quadrant is close to the % of the signal (5%)                                                   
                                    counts[signalName.index(name)] = counts[signalName.index(name)] + 1                             # Add 1 to the count    
    return signalName[counts.index(max(counts))]                                                                                    # Return the name of the signal with the highest probability


dataBaseFile = open("Projeto/TrafficSignalDetection/trafficSignDataBase.json")
dataBase = json.load(dataBaseFile)

for img in trafficSignals:
    print(detectSignal(img))
    cv.imshow("Signal", img)
    cv.waitKey(0)

dataBaseFile.close()
cv.destroyAllWindows()
