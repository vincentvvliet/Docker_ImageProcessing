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


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


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
    # Apply yellow mask
    mask = apply_yellow_mask(image)

    # plotImage(mask)

    # Convert to grayscale and apply gaussian
    gray = apply_gaussian(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

    # Canny
    edges = cv2.Canny(gray, 100, 200)

    # Remove noise and connect rectangle that encompasses license plate
    edges = cv2.erode(cv2.dilate(edges, np.ones((3, 3))), np.ones((3, 3)))

    # Get all contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # TODO add checks
    # 0 - 20 contours, select only licence plate
    # Must be rectangle with certain ratio's
    # if less than 1-2% of image size, then discard (noise)
    # if more than 20-30%, then discard (too big)
    # cv2.minAreaRect(c)
    # if area contour < 50% of whole rectangle size, since no rotation bigger than specific rotation, discard (too large rotation)
    # for minarearect, width height -> if aspect ratio not between [3-7.5] (change value), then discard (wrong ratio)
    # for some images, more than 1 licence plate passes these checks

    count = 0
    # some variables for further use
    center = (int(len(image[0]) / 2), int(len(image) / 2))
    ratio = float(47 / 11)
    differences = {}
    boxes = []
    rotate_matrices = []

    # choose the best contour
    for c in contours:

        # skip the ones with less than 200 pixels or greater than 20% of image size
        if len(c) < 200:  # or len(c) < 0.02 * len(image) or len(c) > 0.2 * len(image):
            continue

        # DEBUGGING
        # img_contours = np.zeros(image.shape)
        # cv2.drawContours(img_contours, c, -1, (0, 255, 0), 3)
        # plotImage(img_contours)
        # print("this:", c.shape)
        # print(image.shape)
        # print("if less than:", 0.02 * len(image) * len(image[0]))
        # print("if greater than:", 0.2 * len(image) * len(image[0]))

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
        ratio2 = width / height

        if width < 90 or height < 20 or ratio2 < 2:
            continue

        r = cv2.warpAffine(image, M, (len(image[0]), len(image)))
        # plotImage(r[mini:maxi, minj:maxj])

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
    return rotated[final_box[0]:final_box[1] + 1, final_box[2]:final_box[3] + 1], True


def apply_yellow_mask(image):
    # Define color range
    colorMin = np.array([6, 100, 100])
    colorMax = np.array([30, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    masked = cv2.bitwise_and(image, image, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    return masked


def apply_gaussian(image):
    return cv2.filter2D(image, -1, cv2.getGaussianKernel(5, 5))
