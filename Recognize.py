import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

AMBIGUOUS_RESULT = "AMBIGUOUS"
EPSILON = 0.15
# Load the reference characters
character_set = {'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                 '3', '4', '5', '6', '7', '8', '9'}
# letter_set = {'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z'}
# number_set = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
reference_characters = {}
path = "TrainingSet/Categorie I/"


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def loadImage(filepath, filename, grayscale=True):
    return cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def difference_score(test_image, reference_character):
    # return the number of non-zero pixels
    return np.count_nonzero(cv2.bitwise_xor(test_image, reference_character))


def give_label_two_scores(test_image):
    # Get the difference score with each of the reference characters
    difference_scores = []
    for char in reference_characters:
        difference_scores.append(difference_score(test_image, reference_characters[char]))

    difference_scores = np.array(difference_scores)
    A, B = np.partition(difference_scores, 1)[0:2]
    result_char = 0

    # Check if the ratio of the two scores is close to 1 (if so return AMBIGUOUS_RESULT)
    for char in reference_characters:
        if difference_score(test_image, reference_characters[char]) == A:
            result_char = char

    ratio = A / B
    if ratio > 1 - EPSILON and ratio < 1 + EPSILON:
        return AMBIGUOUS_RESULT
    # Return a single character based on the lowest score
    return result_char


"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""


def segment_and_recognize(plate_imgs):
    recognized_plates = []
    for image in plate_imgs:
        # TODO crop image further to separate each letter & number
        plotImage(image, give_label_two_scores(image))
        if give_label_two_scores(image) != AMBIGUOUS_RESULT:
            recognized_plates.append(image)

    return recognized_plates


def setup():
    # Setup reference characters
    letter_counter = 0
    number_counter = 0
    for char in character_set:
        if char.isdigit():
            reference_characters[char] = loadImage("SameSizeLetters/", str(number_counter) + ".bmp")
            number_counter = number_counter + 1
        else:
            reference_characters[char] = loadImage("SameSizeNumbers/", str(letter_counter) + ".bmp")
            letter_counter = letter_counter + 1

    # TODO remove when done debugging
    # Capture frame with license plate
    cap = cv2.VideoCapture(path + "Video2_2.avi")

    # Choose a frame to work on
    frameN = 36
    frame = 0

    for i in range(0, frameN):
        # Read the video frame by frame
        ret, frame = cap.read()
        print(ret)
        # if we have no more frames end the loop
        if not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plotImage(frame, "Frame")

    # test_images = []
    # test_images.append(frame)
    # segment_and_recognize(test_images)

    # Define color range
    colorMin = np.array([10, 120, 100])
    # colorMin = np.array([14, 110, 96])
    colorMax = np.array([30, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    # Plot the masked image (where only the selected color is visible)
    # plotImage(mask, "Masked image", "gray")
    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

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
    plotImage(cropped, "Cropped")

setup()

# for image in plate_imgs:
#     plotImage(image, give_label_two_scores(image))
# for i in range(2, 8):
#     image_name = "Video" + str(i) + "_2.avi"
#     test_images.append(loadImage("TrainingSet/Categorie I", image_name))
