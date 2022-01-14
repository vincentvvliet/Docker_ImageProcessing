import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np

AMBIGUOUS_RESULT = "AMBIGUOUS"
EPSILON = 0.15
# Load the reference characters
character_array = ['B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                   '3', '4', '5', '6', '7', '8', '9']

reference_characters = {}
path = "TrainingSet/Categorie I/"
recognized_plates = []


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def loadImage(filepath, filename, grayscale=True):
    return cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def difference_score(test_image, reference_character):
    reference_character = cv2.resize(reference_character, (len(test_image[0]), len(test_image)))
    # return the number of non-zero pixels
    return np.count_nonzero(cv2.bitwise_xor(test_image, reference_character))


def give_label_two_scores(test_image):
    # Get the difference score with each of the reference characters

    # Erode to remove noise
    test_image = cv2.erode(test_image, np.ones((2, 2)))

    difference_scores = []
    for key, value in reference_characters.items():
        # Debug scores
        # print("key",key,"score:",difference_score(test_image, value))
        difference_scores.append(difference_score(test_image, value))

    difference_scores = np.array(difference_scores)
    A, B = np.partition(difference_scores, 1)[0:2]
    result_char = 0

    # Check if the ratio of the two scores is close to 1 (if so return AMBIGUOUS_RESULT)
    for key, value in reference_characters.items():
        if difference_score(test_image, value) == A:
            # Debug which reference is closest to image
            # plotImage(test_image)
            # plotImage(value)
            result_char = key

    # ratio = A / B
    # print("ratio:", ratio)
    # if ratio > 1 - EPSILON and ratio < 1 + EPSILON:
    #     return AMBIGUOUS_RESULT

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


def segment_and_recognize(plate_imgs,frame):
    # Call setup only once
    setup()
    # Main functionality
    plate_info = []

    # Add plate characters in correct format
    plate = '\''
    for char in recognize(plate_imgs):
        plate += char
    plate += '\''

    # Append the reconised characters, frame no. and time (rounded down)
    plate_info.append(plate)
    plate_info.append(frame)
    plate_info.append(round(frame / 24))

    # Add to list of known plates
    recognized_plates.append(plate_info)

    # Write to csv
    write(recognized_plates)


def recognize(plate_imgs):
    recognized_chars = []
    images = seperate(plate_imgs)
    for image in images:
        character = give_label_two_scores(image)
        if character != AMBIGUOUS_RESULT:
            recognized_chars.append(character)
    # print("recognized:", recognized_chars)

    return recognized_chars


def setup():
    # Setup reference characters
    letter_counter = 1  # starts at 1.bmp
    number_counter = 0
    for char in character_array:
        if char.isdigit():
            reference_characters[char] = loadImage("SameSizeNumbers/", str(number_counter) + ".bmp")
            number_counter = number_counter + 1
        else:
            reference_characters[char] = loadImage("SameSizeLetters/", str(letter_counter) + ".bmp")
            letter_counter = letter_counter + 1

    # Resize reference characters
    for char, value in reference_characters.items():
        reference_characters[char] = crop_to_boundingbox(value)


def write(plates):
    # open the file in the write mode
    with open('sampleOutput.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        header = ['License plate','Frame no.','Timestamp(seconds)']

        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)


def crop_to_boundingbox(image):
    # plotImage(image)
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
    return image[mini:maxi + 1, minj:maxj + 1]


def seperate(image):
    # plotImage(image, "")
    # convert to grayscale
    plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # use mean of all colors as threshold and convert to binary where characters are white and background black
    threshold = np.mean(plate)
    for i in range(len(plate)):
        for j in range(len(plate[0])):
            if plate[i][j] < threshold:
                plate[i][j] = 255
            else:
                plate[i][j] = 0

    # use epsilon, this is
    epsilon = 0.05 * len(plate[0])

    # after observing multiple plates, I saw that each character has a width of approximately 10% of the plate's width
    charwidth = 0.1 * len(plate[0])

    boxes = []
    overlap = np.array([])
    while (len(boxes) < 6 and len(overlap) < len(plate[0]) - int(charwidth)):
        minwhite = 2 * len(plate)
        box = (0, 0)
        for j in range(int(len(plate[0]) - charwidth - epsilon)):
            if j not in overlap:
                whites = cv2.countNonZero(plate[:, j])
                for jj in range(int(j + charwidth), int(j + charwidth + epsilon)):
                    if (not np.in1d(overlap, np.arange(j, jj)).any()) and whites + cv2.countNonZero(
                            plate[:, jj]) < minwhite:
                        minwhite = whites + cv2.countNonZero(plate[:, jj])
                        box = (j, jj)
        overlap = np.concatenate((overlap, np.arange(box[0], box[1] + 1)))
        boxes.append(box)

    boxes.sort()
    gaps = []
    for i in range(len(boxes) - 1):
        gaps.append(boxes[i + 1][0] - boxes[i][1])
    gaps.sort()
    gap1 = gaps[-1]
    gap2 = gaps[-2]

    dot1 = (0, 0)
    dot2 = (0, 0)
    for i in range(len(boxes) - 1):
        gap = boxes[i + 1][0] - boxes[i][1]
        if dot1 == (0, 0) and gap1 == gap:
            dot1 = (boxes[i + 1][0], boxes[i][1])
        elif dot2 == (0, 0) and gap2 == gap:
            dot2 = (boxes[i + 1][0], boxes[i][1])

    dot1index = int((dot1[0] + dot1[1]) / 2)
    dot2index = int((dot2[0] + dot2[1]) / 2)

    black = True
    matches = []
    for i in range(len(plate)):
        if plate[i][dot1index] == 255 and plate[i][dot2index] == 255:
            if black:
                matches.append([i])
                black = False
        else:
            if not black:
                matches[-1].append(i)
                black = True

    finalindex = int(len(plate) / 2)
    if len(matches) > 0:
        if len(matches[-1]) == 1:
            matches[-1].append(len(plate))
        center = float(len(plate) / 2)
        mindis = len(plate)
        for m in matches:
            if np.abs(center - (float((m[1] + m[0]) / 2.0))) < mindis:
                mindis = np.abs(center - (float((m[1] + m[0]) / 2.0)))
                finalindex = int((m[1] + m[0]) / 2)

    heightchar = float(0.17 * len(plate[0]))
    if finalindex < float(heightchar / 2.0) or finalindex > len(plate) - float(heightchar / 2.0):
        finalindex = int(len(plate) / 2)

    finalindex += int(0.02 * len(plate))
    ymin = finalindex - int(heightchar / 2)
    ymax = finalindex + int(heightchar / 2)

    characters = []
    for box in boxes:
        char = plate[ymin:ymax, box[0]:box[1]]
        # eventueel dilate en erode om mooier te maken
        char = crop_to_boundingbox(char)
        characters.append(char)
    return characters

    # for i in range(len(image)):
    #     for j in range(len(image[0])):
    #         if plate[i][j] == 255:
    #             image[i][j] = [255,255,255]
    #         else:
    #             image[i][j] = [0,0,0]

    # image[:,dot1index] = [0,255,0]
    # image[:,dot2index] = [0,255,0]
    # plotImage(image, "")

    # mark found positions green
    # for box in boxes:
    #     image[0:3,box[0]:box[1]+1] = [0,255,0]
    # plotImage(image, "")

    # for box in boxes:
    #     char = plate[:,box[0]:box[1]]
    #     contours, hierarchy = cv2.findContours(char,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #     biggest = contours[0]
    #     for c in contours:    
    #         if len(c) > len(biggest):
    #             biggest = c
    #     pixels = np.zeros((len(char),len(char[0])))
    #     imin = len(char)
    #     imax = 0
    #     jmin = len(char[0])
    #     jmax = 0
    #     for p in biggest:
    #         pixels[p[0][1]][p[0][0]] = 255
    #         if p[0][1] < imin:
    #             imin = p[0][1]
    #         if p[0][1] > imax:
    #             imax = p[0][1]
    #         if p[0][0] < jmin:
    #             jmin = p[0][0]
    #         if p[0][0] > jmax:
    #             jmax = p[0][0]
    #     plotImage(char[imin:imax,jmin:jmax], "")

    # width = 0.9*len(plate[0])
    # for i in range(1, 6):
    #     minwhite = len(plate)
    #     split = 0
    #     mincolumn = int((i*width/6)-0.07*width)
    #     maxcolumn = int((i*width/6)+0.07*width)
    #     for j in range(mincolumn, maxcolumn):
    #         column = plate[:,j]
    #         whites = cv2.countNonZero(column)
    #         if whites < minwhite:
    #             minwhite = whites
    #             split = j
    #     image[:, split] = [0, 255, 0]
    # plotImage(image, "")

    # contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # for c in contours:    
    #     pixels = np.zeros((len(image),len(image[0])))
    #     for p in c:
    #         pixels[p[0][1]][p[0][0]] = 255
    #     plotImage(pixels, "")

    # ret2,th2 = cv2.threshold(plate,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plotImage(th2, "thresh")

    #
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
