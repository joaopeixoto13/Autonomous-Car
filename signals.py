import cv2 as cv
import numpy as np

WIDTH = 50
HEIGHT = 50

WINDOW_WIDTH = 512
WINDOW_HEIGHT = 256

""" finish_signal = cv.imread("Images/finish.png")
park_signal = cv.imread("Images/park.png")
right_signal = cv.imread("Images/right.png")
stop_signal = cv.imread("Images/stop.png")
ahead_signal = cv.imread("Images/ahead.png")
left_signal = cv.imread("Images/left.png")

finish_signal = cv.resize(finish_signal, (WIDTH, HEIGHT))
park_signal = cv.resize(park_signal, (WIDTH, HEIGHT))
right_signal = cv.resize(right_signal, (WIDTH, HEIGHT))
stop_signal = cv.resize(stop_signal, (WIDTH, HEIGHT))
ahead_signal = cv.resize(ahead_signal, (WIDTH, HEIGHT))
left_signal = cv.resize(left_signal, (WIDTH, HEIGHT))

image_1 = cv.imread("Images/1.png")
image_2 = cv.imread("Images/2.png")
image_3 = cv.imread("Images/3.png")
image_4 = cv.imread("Images/4.png")
image_5 = cv.imread("Images/5.png")

image_1 = cv.resize(image_1, (WINDOW_WIDTH, WINDOW_HEIGHT))
image_2 = cv.resize(image_2, (WINDOW_WIDTH, WINDOW_HEIGHT))
image_3 = cv.resize(image_3, (WINDOW_WIDTH, WINDOW_HEIGHT))
image_4 = cv.resize(image_4, (WINDOW_WIDTH, WINDOW_HEIGHT))
image_5 = cv.resize(image_5, (WINDOW_WIDTH, WINDOW_HEIGHT)) 

signals = [left_signal, ahead_signal, right_signal, stop_signal, park_signal, finish_signal]
images = [image_1, image_2, image_3, image_4, image_5]  """

# BGR
boundaries = [
    ([0, 230, 0], [25, 255, 25]),       # green
    ([0, 0, 230], [25, 25, 255]),       # red
    ([0, 200, 220], [25, 255, 255]),    # yellow
]

def separateColors(img):
    output = []
    for lower, upper in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv.inRange(img, lower, upper)
        output.append(cv.bitwise_and(img, img, mask=mask))
    return output

def isolateSignal(img):
    output = separateColors(img)
    final_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(0, len(output)):
        final_img = cv.bitwise_or(final_img, output[i])
    return final_img

def detectSignal(img):
    max_th = 5
    min_th = 25
    stretch_x_factor = 7
    stretch_y_factor = 1
    arrow_th = 4
    img = isolateSignal(img)
    b, g, r = cv.split(img)
    b, g, r = b.sum()//255, g.sum()//255, r.sum()//255

    # Park or Finish
    if g > b*max_th and r > b*max_th:
        output = separateColors(img)
        if output[1].sum() > min_th:
            return "finish"
        elif output[2].sum() > min_th:
            return "park"
        else:
            return "none"
    elif r > b*max_th and r > g*max_th: 
        return "stop"
    elif g > b*max_th and g > r*max_th:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = cv.resize(gray_img, (0, 0), fx=stretch_x_factor, fy=stretch_y_factor)
        _, gray_img = cv.threshold(gray_img, 50, 255, cv.THRESH_BINARY)
        pos = np.nonzero(gray_img)
        xmin = pos[1].min()
        xmax = pos[1].max()
        ymin = pos[0].min()
        ymax = pos[0].max()
        Xc = (xmin + xmax) // 2
        Yc = (ymin + ymax) // 2

        M = cv.moments(gray_img)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if cX > Xc + arrow_th:
            return "right"
        elif cX < Xc - arrow_th:
            return "left"
        else:
            return "ahead"
    else:
        return "none"


""" for img in images:
    print(detectSignal(img))
    cv.imshow("Image", img)
    cv.waitKey(0)
cv.destroyAllWindows() """