import cv2 as cv
import numpy as np
from utils import stackImages

WIDTH = 256
HEIGHT = 256

# BGR
boundaries = [
    ([0, 230, 0], [25, 255, 25]),       # Green detection boundaries
    ([0, 0, 230], [25, 25, 255]),       # Red detection boundaries
    ([0, 200, 220], [25, 255, 255]),    # Yellow detection boundaries
    ([240, 240, 240], [255, 255, 255]), # White detection boundaries
    ([0, 0, 0], [15, 15, 15])           # Black detection boundaries
]

""" RED_HUE_1 = (330, 360)
RED_HUE_2 = (0, 30)
GREEN_HUE = (90, 140)
BLUE_HUE = (150,250)
YELLOW_HUE = (35, 80)


boundaries = [
    ([RED_HUE_1[0], 0, 0], [RED_HUE_1[1], 255, 180]),           # Red detection boundaries
    ([RED_HUE_2[0], 0, 0], [RED_HUE_2[1], 255, 180]),           # Red detection boundaries
    ([GREEN_HUE[0], 0, 0], [GREEN_HUE[1], 255, 180]),           # Green detection boundaries
    ([BLUE_HUE[0], 0, 0], [BLUE_HUE[1], 255, 180]),             # Blue detection boundaries
    ([YELLOW_HUE[0], 0, 0], [YELLOW_HUE[1], 255, 180]),         # Yellow detection boundaries
    ([0, 0, 0], [255, 255, 180])                                # White detection boundaries
    ([0, 0, 0], [255, 255, 180])                                # Black detection boundaries
] """

def separateColors(img):
    output = [] 
    for lower, upper in boundaries:                         # For each boundarie
        lower = np.array(lower, dtype="uint8")              # Get the lower boundarie and convert to numpy array
        upper = np.array(upper, dtype="uint8")              # Get the upper boundarie and convert to numpy array

        mask = cv.inRange(img, lower, upper)                # Create a mask with the boundarie
        output.append(cv.bitwise_and(img, img, mask=mask))  # Apply the mask to the image
    return output                                           # Return the list of images

def isolateSignal(img): 
    output = separateColors(img)                                            # Separate the colors
    final_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)         # Create a black image
    for i in range(0, len(output)):                                         # For each color
        final_img = cv.bitwise_or(final_img, output[i])                     # Add the color to the final image
    return final_img                                                        # Return the final image

def imgProcess(img):
    img = cv.resize(img, (WIDTH, HEIGHT))
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    #img_stacked = stackImages(0.5, [[img_red, img_green], [img_blue, img_yellow]])
    return img


img = cv.imread("Images/1.png")
img = imgProcess(img)
#img = isolateSignal(img)
cv.imshow("Img", img)
cv.waitKey(0)
cv.destroyAllWindows()