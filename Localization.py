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
    """Image plotting for debugging."""
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def find_plate(image):
    """Localization pipeline."""

    # Apply yellow mask
    mask = apply_yellow_mask(image)

    # plotImage(mask)

    # Convert to grayscale and apply gaussian
    gray = apply_gaussian(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

    # Get all contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # some variables for further use
    count = 0
    ratio = float(47 / 11)
    differences = {}
    angles = []
    centers = []
    heights = []
    widths = []

    # choose the best contour
    for c in contours:
        area = cv2.contourArea(c)
        total_area = image.shape[0] * image.shape[1]

        # skip the contours that match the following:
        # less than 200 pixels
        # less than 0.5% of image size (too small to be a plate)
        # greater than 20% of image size (too large to be a plate)
        if len(c) < 200 or area < 0.005 * total_area or area > 0.2 * total_area:
            continue

        # get the orientation angle and rotate the contour
        (center, (width, height), angle) = cv2.minAreaRect(c)

        if angle < -45:
            width, height = height, width
            angle = angle + 90

        if width < 90 or height < 20 or width / height < 2:
            continue

        diff = float(np.abs(float(ratio - float(width / height))))
        differences[count] = diff
        angles.append(angle)
        centers.append(center)
        heights.append(height)
        widths.append(width)

        count += 1

    if len(differences) < 1:
        return np.array([]), False

    chosen = min(differences, key=differences.get)
    angle = angles[chosen]
    center = centers[chosen]
    height = heights[chosen]
    width = widths[chosen]

    rotated = rotate(image, angle, center)
    result = rotated[int(center[1] - 0.5 * height):int(center[1] + 0.5 * height),
             int(center[0] - 0.5 * width):int(center[0] + 0.5 * width)]

    return result, True


def apply_yellow_mask(image):
    """Apply mask of color range that fits the yellow licence plates."""

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
    """Apply gaussian filtering."""

    return cv2.filter2D(image, -1, cv2.getGaussianKernel(5, 5))


def rotate(image, angle, center):
    (h, w) = image.shape[:2]

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
