import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def plate_detection(image):
    plate_imgs = image

    # Define color range
    colorMin = np.array([10, 100, 100])
    colorMax = np.array([20, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    # Plot the masked image (where only the selected color is visible)
    # plotImage(mask, "Masked image", "gray")
    result = cv2.bitwise_and(plate_imgs, plate_imgs, mask=mask)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # plotImage(result, "Result")

    # Get coordinates of the plate
    indices = []
    for i, _ in enumerate(mask):
        for j, _ in enumerate(mask[i]):
            if mask[i][j] != 0:
                indices.append([i, j])
    indices = sorted(indices, key=lambda indices: indices[0])
    minXY = indices[0]
    maxXY = indices[-1]
    cropped = result[minXY[0]:maxXY[0], minXY[1]:maxXY[1]]
    # plotImage(cropped, "Cropped")

    return cropped


def plotImage(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()
