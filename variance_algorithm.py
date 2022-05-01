from cmath import isnan, pi
import cv2 as cv
import numpy as np
import math

WIDTH = 640
HEIGHT = 480
CENTER_COORD = (WIDTH//2,HEIGHT-1)

CENTER_COORD_1 = (WIDTH//4, HEIGHT - 1)
CENTER_COORD_2 =(3*WIDTH//4, HEIGHT - 1)

font = cv.FONT_HERSHEY_COMPLEX

def ImgProcess(img):
    img = cv.resize(img, (WIDTH,HEIGHT))
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    img = cv.blur(img,(5,5))
    img = cv.Canny(img,100,200)
    return img

def drawLines(img, center, start_angle, end_angle):
    for i in range (start_angle,end_angle+1,8):
        CalcXFinal = center[0] + (WIDTH-350) * math.cos(math.radians(i))
        CalcYFinal = center[1] - (WIDTH-350) * math.sin(math.radians(i))
        #CalcXInit = center[0] + HEIGHT//4 * math.cos(math.radians(i))
        #CalcYInit = center[1] - HEIGHT//4 * math.sin(math.radians(i)) 
        cv.line(img,(int(center[0]),int(center[1])),(int(CalcXFinal),int(CalcYFinal)),color = 255, thickness= 1)

def drawLidar(img, center, start_radius, end_radius):
    for i in range (start_radius,end_radius+1,30):
        cv.ellipse(img,center,(i,i),0,15,165,color = 255, thickness= 1)
    return img

def getContours(img):
    merged_list = []
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = np.asarray(contours, dtype=object)
    for l in contours:
        for i in l:
            for j in i:
                merged_list.append(j)

    return merged_list

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

def filterExtreme(points):
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    x_min = x_mean - 2 * x_std
    x_max = x_mean + 2 * x_std
    y_min = y_mean - 2 * y_std
    y_max = y_mean + 2 * y_std
    filtered_points = []
    for i in range(0,len(points)):
        if(points[i][0] > x_min and points[i][0] < x_max and points[i][1] > y_min and points[i][1] < y_max):
            filtered_points.append(points[i])
    return filtered_points

def control(pointsl,pointsr):
    global u_array
    xl = [i[0] for i in pointsl]
    xr = [i[0] for i in pointsr]
    if (len(xl) == 0 or len(xr) == 0):
        return 0
    xl_mean = np.mean(xl)
    xr_mean = np.mean(xr)
    xl_std = np.std(xl)
    xr_std = np.std(xr)
    
    std = (xl_std - xr_std)
    mean = ((xl_mean + xr_mean + 370) / 2) - 320
    u = 0.70*std + 0.30*mean                    
    u_array = np.roll(u_array, -1, axis=0)
    u_array[4] = u
    u_array = np.multiply(u_array, n_array)
    u_mean = np.sum(u_array)
    return u_mean 

def controlProcess(u):
    if (u > 60):
        u = 60
    elif (u < -60):
        u = -60
    u = u * Kp
    u = u * -1      # por causa do simulador
    return u    


def twoLineAlgorithm(img, flag):
    global u
    global u_prev
    global u_array
    global n_array
    u_prev = u
    if (flag):
        u_array[0] = u_array[1] = u_array[2] = u_array[3] = u_array[4] = 0
    imgA = ImgProcess(img)
    #imgB2 = imgA.copy()
    imgB = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
    drawLidar(imgB, (WIDTH//2,HEIGHT//3), 0, 450)
    #drawLidar(imgB2, (WIDTH//2,HEIGHT//3), 0, 450)
    imgC = cv.bitwise_and(imgA, imgB) # Apply the mask
    left =  np.asarray(getContours(imgC[:,:WIDTH//2 - 50]), dtype=object)
    right =  np.asarray(getContours(imgC[:,WIDTH//2 + 50:WIDTH]), dtype=object)
    if len(right) > 0 and len(left) > 0:
        u = control(filterExtreme(left),filterExtreme(right))
    else:
        print("No Line --> In Front")
        u = u_prev
    #cv.imshow("img",img)
    u = controlProcess(u)
    return u

n_array = np.array([0.05, 0.1, 0.15, 0.20, 0.5])
u_array =  np.array([0,0,0,0,0], dtype=float)

u = 0
u_prev = 0
Kp = 0.75

