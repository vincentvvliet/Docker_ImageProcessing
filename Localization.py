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

def get_bounding_box(image):
    mini = len(image)
    minj = len(image[0])
    maxi = 0
    maxj = 0
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] != 0:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                if j > maxj:
                    maxj = j
    return mini, maxi, minj, maxj

def find_plate(image):
    # apply yellow mask
    mask = apply_yellow_mask(image)

    # make grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # get all contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    count = 0
    # some variables for further use
    center = (int(len(image[0]) / 2), int(len(image) / 2))
    ratio = float(47 / 11)
    differences = {}
    boxes = []
    rotate_matrices = []

    # choose the best contour
    for c in contours:

        # skip the ones with less than 200 pixels
        if len(c) < 200:
            continue

        # store the pixels of current contour
        pixels = np.zeros((len(image), len(image[0])))
        for pixel in c:
            i = pixel[0][1]
            j = pixel[0][0]
            pixels[i][j] = 255

        # get the orientation angle and rotate the contour
        arr = cv2.minAreaRect(c)
        angle = arr[-1]
        if angle < -45:
            angle = angle + 90
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(pixels, M, (len(image[0]), len(image)))

        # get boundary box of this rotated version and calculate the height/width ratio
        mini, maxi, minj, maxj = get_bounding_box(rotated)
        width = maxj - minj
        height = maxi - mini
        if width < 90 or height < 20:
            continue

        diff = float(np.abs(float(ratio - float(width / height))))
        differences[count] = diff
        rotate_matrices.append(M)
        boxes.append((mini, maxi, minj, maxj))
        count += 1

    if len(differences) < 1:
        return np.array([]), False

    chosen = min(differences, key=differences.get)
    final_M = rotate_matrices[chosen]
    final_box = boxes[chosen]
    rotated = cv2.warpAffine(image, final_M, (len(image[0]), len(image)))
    return rotated[final_box[0]:final_box[1]+1,final_box[2]:final_box[3]+1], True


def apply_yellow_mask(image):
    image = apply_gaussian(image)
    # Define color range
    colorMin = np.array([1, 60, 60])
    colorMax = np.array([26, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    masked = cv2.bitwise_and(image, image, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    return masked

def apply_gaussian(image):
    return cv2.filter2D(image, -1, cv2.getGaussianKernel(30, 10))