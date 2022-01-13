import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from Localization import get_plate

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
    cap = cv2.VideoCapture(path + "Video1_2.avi")
    
    # Choose a frame to work on
    frameN = 36
    frame = 0
    
    for i in range(0, frameN):
        # Read the video frame by frame
        ret, frame = cap.read()
        # if we have no more frames end the loop
        if not ret:
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    plate = get_plate(frame)

    # test_images = []
    # test_images.append(frame)
    # segment_and_recognize(test_images)
    
def seperate(image):
    colorMin = np.array([10, 60, 60])
    colorMax = np.array([26, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    # hsi = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsi, colorMin, colorMax)
    # masked = cv2.bitwise_and(plate, plate, mask=mask)
    # masked = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plate = cv2.erode(plate, (550,550))
    plate = cv2.dilate(plate, (20,20))
    threshold = np.mean(plate)
    for i in range(len(plate)):
        for j in range(len(plate[0])):
            if plate[i][j] < threshold:
                plate[i][j] = 255
            else:
                plate[i][j] = 0

    plotImage(plate, "")


    first = plate[:,:int(len(plate[0])/6)]



    # contours, hierarchy = cv2.findContours(first,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # biggest = contours[0]
    # for c in contours:
    #     if len(c) > len(biggest):
    #         biggest = c
        # imin = len(plate)
        # imax = 0
        # jmin = len(plate[0])
        # jmax = 0
        # pixels = np.zeros((len(plate),len(plate[0])))
        # for p in c:
        #     pixels[p[0][1]][p[0][0]] = 255
        #     if p[0][1] < imin:
        #         imin = p[0][1]
        #     if p[0][1] > imax:
        #         imax = p[0][1]
        #     if p[0][0] < jmin:
        #         jmin = p[0][0]
        #     if p[0][0] > jmax:
        #         jmax = p[0][0]
        #     if imax - imin < 0.25*len(plate) or jmax - jmin < 0.025*len(plate[0]):
        #         plate[imin:imax,jmin:jmax] = 0

# setup()
# for image in plate_imgs:
#     plotImage(image, give_label_two_scores(image))
# for i in range(2, 8):
#     image_name = "Video" + str(i) + "_2.avi"
#     test_images.append(loadImage("TrainingSet/Categorie I", image_name))
