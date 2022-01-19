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

def plotImage(img, title="", cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


# TODO not used
def get_orientation_distribution(fourier):
    N = fourier.shape[0]
    M = fourier.shape[1]
    orientation = np.zeros(360)
    for i in range(fourier.shape[0]):
        for j in range(fourier.shape[1]):
            u_r = np.sqrt((i - N / 2) ** 2 + (j - M / 2) ** 2)
            theta = np.arctan((i - N / 2) / (j - M / 2)) if (j - M / 2) != 0 else 0
            orientation[int(np.rad2deg(theta))] += np.abs(fourier[int(u_r * np.sin(theta)), int(u_r * np.cos(theta))])
    return orientation


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

def get_all_contours_info(image):
    # apply yellow mask
    mask = apply_yellow_mask(image)

    # make grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # get all contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    differences = []
    boxes = []
    angles = []
    count = 0
    # some variables for further use
    center = (int(len(image[0]) / 2), int(len(image) / 2))
    ratio = float(47 / 11)

    # choose the best contour
    for c in contours:

        # skip the ones with less than 20 pixels
        if len(c) < 40:
            differences.append(float('inf'))
            boxes.append((0,0,0,0))
            angles.append(0)
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
        if width < 2 or height < 2:
            differences.append(float('inf'))
            boxes.append((0,0,0,0))
            angles.append(0)
            continue

        diff = float(np.abs(float(ratio - float(width / height))))
        differences.append(diff)
        boxes.append((mini, maxi, minj, maxj))
        angles.append(angle)
        count += 1

    differences = np.argsort(np.array(differences))[0:4]
    return differences[0:count], angles, boxes


def get_plate(image):
    # apply yellow mask
    mask = apply_yellow_mask(image)

    # make grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # get all contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # some variables for further use
    center = (int(len(image[0]) / 2), int(len(image) / 2))
    ratio = float(47 / 11)
    finalbox = 0, 0, 0, 0
    finalangle = 0
    difference = float('inf')

    # choose the best contour
    for c in contours:

        # skip the ones with less than 20 pixels
        if len(c) < 40:
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
        if width < 2 or height < 2:
            continue

        diff = float(np.abs(float(ratio - float(width / height))))
        if diff < difference:
            difference = diff
            finalbox = mini, maxi, minj, maxj
            finalangle = angle

    # rotate the original image with the angle of the chosen contour
    M = cv2.getRotationMatrix2D(center, finalangle, 1.0)
    rotated = cv2.warpAffine(image, M, (len(image[0]), len(image)))

    # crop the rotated image with the boundary points of the chosen contour
    crop = rotated[finalbox[0]:finalbox[1], finalbox[2]:finalbox[3]]
    return crop




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

# TODO not used
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
    kernel = cv2.getGaussianKernel(30, 10)
    image = cv2.filter2D(image, -1, kernel)
    # Define color range
    colorMin = np.array([10, 60, 60])
    colorMax = np.array([26, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    masked = cv2.bitwise_and(image, image, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    return masked


# TODO not used
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


# TODO not used
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


# TODO not used
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


def loadImage(filepath, filename, grayscale=True):
    return cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
