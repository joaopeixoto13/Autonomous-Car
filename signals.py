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
    ([0, 230, 0], [25, 255, 25]),                                           # Green detection boundaries
    ([0, 0, 230], [25, 25, 255]),                                           # Red detection boundaries
    ([0, 200, 220], [25, 255, 255]),                                        # Yellow detection boundaries
]

def separateColors(img):
    output = [] 
    for lower, upper in boundaries:                                         # For each boundarie
        lower = np.array(lower, dtype="uint8")                              # Get the lower boundarie and convert to numpy array
        upper = np.array(upper, dtype="uint8")                              # Get the upper boundarie and convert to numpy array

        mask = cv.inRange(img, lower, upper)                                # Create a mask with the boundarie
        output.append(cv.bitwise_and(img, img, mask=mask))                  # Apply the mask to the image
    return output                                                           # Return the list of images

def isolateSignal(img): 
    output = separateColors(img)                                            # Separate the colors
    final_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)         # Create a black image
    for i in range(0, len(output)):                                         # For each color
        final_img = cv.bitwise_or(final_img, output[i])                     # Add the color to the final image
    return final_img                                                        # Return the final image

def detectSignal(img):
    max_th = 5                                                              # Maximum threshold
    min_th = 25                                                             # Minimum threshold
    stretch_x_factor = 7                                                    # Stretch factor in x direction
    stretch_y_factor = 1                                                    # Stretch factor in y direction
    arrow_th = 4                                                            # Threshold for the arrow
    img = isolateSignal(img)                                                # Isolate the signal
    b, g, r = cv.split(img)                                                 # Split the image into BGR
    b, g, r = b.sum()//255, g.sum()//255, r.sum()//255                      # Rescale each color
 
    # Park or Finish
    if g > b*max_th and r > b*max_th:                                       # If the green and red are much greater than the blue
        output = separateColors(img)                                        # Separate the colors
        if output[1].sum() > min_th:                                        # If the red is greater than the minimum threshold
            return "finish"                                                 # Return finish
        elif output[2].sum() > min_th:                                      # If the yellow is greater than the minimum threshold
            return "park"                                                   # Return park
        else:                                                               # If none of the above
            return "none"                                                   # Return none (error case)
    elif r > b*max_th and r > g*max_th:                                     # If the red is much greater than the blue and green
        return "stop"                                                       # Return stop
    elif g > b*max_th and g > r*max_th:                                     # If the green is much greater than the blue and red
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                      # Convert the image to grayscale
        gray_img = cv.resize(gray_img, (0, 0), fx=stretch_x_factor, fy=stretch_y_factor)    # Stretch the image
        _, gray_img = cv.threshold(gray_img, 50, 255, cv.THRESH_BINARY)     # Threshold the image
        pos = np.nonzero(gray_img)                                          # Get the nonzero values
        xmin = pos[1].min()                                                 # Get the minimum x value
        xmax = pos[1].max()                                                 # Get the maximum x value
        ymin = pos[0].min()                                                 # Get the minimum y value
        ymax = pos[0].max()                                                 # Get the maximum y value
                                                                            # Calculate the centroid
        Xc = (xmin + xmax) // 2                                             # Get the x center
        Yc = (ymin + ymax) // 2                                             # Get the y center

        M = cv.moments(gray_img)                                            # Get the moments
        cX = int(M["m10"] / M["m00"])                                       # Get the x mass center
        cY = int(M["m01"] / M["m00"])                                       # Get the y mass center

        if cX > Xc + arrow_th:                                              # If the x mass center is greater than the x center + the threshold
            return "right"                                                  # Return right
        elif cX < Xc - arrow_th:                                            # If the x mass center is less than the x center - the threshold
            return "left"                                                   # Return left
        else:                                                               # If the x mass center is less than the x center + the threshold
            return "ahead"                                                  # Return ahead
    else:                                                                   # If none of the above
        return "none"                                                       # Return none (error case)


""" for img in images:
    print(detectSignal(img))
    cv.imshow("Image", img)
    cv.waitKey(0)
cv.destroyAllWindows() """