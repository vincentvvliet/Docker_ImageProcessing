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


def get_boundary_boxes(image):
    # delete some noise 
    image = cv2.erode(image, np.ones((10, 10)))
    image = cv2.dilate(image, np.ones((10, 10)))

    def row_has_white(row):
        for pixel in row:
            if pixel != 0:
                return True
        return False

    # split image horizontal and store both the index of the first row of each box and the index of the last row of each box
    boxes = []
    inWhite = False
    for i in range(0, len(image)):
        row = image[i]
        hasWhite = row_has_white(row)
        if inWhite:
            if not hasWhite:
                boxes[len(boxes) - 1].append(i)
                inWhite = False
        else:
            if hasWhite:
                newBox = []
                newBox.append(i)
                boxes.append(newBox)
                inWhite = True

    # for each box, apply the same technique as above, but now in vertical direction and find the first and last column index
    for box in boxes:
        if len(box) == 1:
            box.append(len(image) - 1)
        inWhite = False
        for j in range(0, len(image[0])):
            allBlack = True
            for i in range(box[0], box[1]):
                if image[i][j] != 0:
                    allBlack = False
                    break
            if inWhite:
                if allBlack:
                    box.append(j)
                    inWhite = False
            else:
                if not allBlack:
                    box.append(j)
                    inWhite = True
        if len(box) == 2:
            box.append(0)
            box.append(0)
        if len(box) % 2 != 0:
            box.append(len(image[0]) - 1)

    # store each box seperate with only four edges
    for box in boxes:
        index = 5
        while index < len(box):
            newBox = [box[0], box[1], box[index - 1], box[index]]
            boxes.append(newBox)
            index += 2

    # crop the new boxes
    for box in boxes:
        inWhite = False
        for i in range(box[0], box[1]):
            row = image[i][box[2]:box[3]]
            hasWhite = row_has_white(row)
            if inWhite:
                if not hasWhite:
                    box[1] = i
                    inWhite = False
            else:
                if hasWhite:
                    box[0] = i
                    inWhite = True

    return boxes


def choose_final_box(boxes):
    # choose boundary box which shape is the closest to the 52cm by 11cm dutch license plate
    ratio = float(52 / 11)
    finalBox = []
    difference = float('inf')
    for box in boxes:
        height = float(box[1] - box[0])
        width = float(box[3] - box[2])
        diff = float(np.abs(float(ratio - float(width / height))))
        if diff < difference:
            difference = diff
            finalBox = [box[0], box[1], box[2], box[3]]
    return finalBox


def apply_yellow_mask(image):
    # Define color range
    colorMin = np.array([10, 60, 60])
    colorMax = np.array([26, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    masked = cv2.bitwise_and(image, image, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    return masked


def plate_detection(image):
    masked = apply_yellow_mask(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # get all the boundary boxes 
    boxes = get_boundary_boxes(gray)
    # choose final boundary box
    box = choose_final_box(boxes)

    if len(box) != 4:
        return np.zeros((len(image), len(image[0])))

    for i in range(0, len(gray)):
        for j in range(0, len(gray[0])):
            if i < box[0] or i > box[1] or j < box[2] or j > box[3]:
                gray[i][j] = 0
            # else:
            #     gray[i][j] = 255
    return gray


def draw_green_box(image):
    masked = apply_yellow_mask(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    boxes = get_boundary_boxes(gray)
    box = choose_final_box(boxes)
    result = image
    if len(box) == 4:
        # color the chosen boundary box green
        for i in range(box[0], box[1]):
            result[i][box[2]] = [0, 255, 0]
            result[i][box[3]] = [0, 255, 0]
        for j in range(box[2], box[3]):
            result[box[0]][j] = [0, 255, 0]
            result[box[1]][j] = [0, 255, 0]

    return result


def draw_all_boxes(image):
    masked = apply_yellow_mask(image)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # get all the boundary boxes 
    boxes = get_boundary_boxes(gray)

    # use same erode and dilate as used in get_boundary_boxes method to better validate the working of this
    gray = cv2.erode(gray, np.ones((10, 10)))
    result = cv2.dilate(gray, np.ones((10, 10)))

    # draw all boxes
    for box in boxes:
        if len(box) == 4:
            for i in range(box[0], box[1]):
                result[i][box[2]] = 255
                result[i][box[3]] = 255
            for j in range(box[2], box[3]):
                result[box[0]][j] = 255
                result[box[1]][j] = 255

    return result


def plotImage(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()
