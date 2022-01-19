import csv

import cv2
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
sift_database = {}
sift = cv2.SIFT_create()
ambiguous = []


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def loadImage(filepath, filename, grayscale=True):
    return cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


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


def segment_and_recognize(image, frame):
    # Call setup only once
    setup()
    # Main functionality
    plate_info = []
    if len(image) < 2 or len(image[0]) < 2:
        return

    binary = make_binary(image)
    char_images, dot1, dot2 = seperate(binary)
    recognized_chars = recognize(char_images, binary)

    recognized = []
    for char in recognized_chars:
        if len(recognized) == dot1 or len(recognized) == dot2:
            recognized.append('-')
        recognized.append(char)
    # print(recognized)
    plate = '\''
    for char in recognized:
        plate += char
    plate += '\''

    # Append the reconised characters, frame no. and time (rounded down)
    plate_info.append(plate)
    plate_info.append(frame)
    plate_info.append(round(frame / 12))

    # Add to list of known plates
    recognized_plates.append(plate_info)

    # Only write at the end
    # if frame > 2000:
        # Write to csv
    write(recognized_plates)


def recognize(images, plate):
    recognized_chars = []
    char_height = int(float(0.17 * len(plate[0])))
    best_score = float('inf')
    for i in range(int(len(plate) - char_height) - 1):
        chars = []
        total_score = 0
        for image in images:
            cropped = image[i:i + char_height]
            cropped = crop_to_boundingbox(cropped)
            if len(cropped) < 2 or len(cropped[0]) < 2:
                total_score = float('inf')
                break
            character, score = give_label_two_scores(cropped)
            total_score += score
            chars.append(character)
        if total_score < best_score:
            best_score = total_score
            recognized_chars = chars

    # recognized_chars = []
    # good = False
    # center = (int(len(image[0]) / 2), int(len(image) / 2))
    # for i in sorted_indices:
    #     M = cv2.getRotationMatrix2D(center, angles[i], 1.0)
    #     rotated = cv2.warpAffine(image, M, (len(image[0]), len(image)))
    #     plate = rotated[boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]]
    #     if len(plate) < 2 or len(plate[0]) < 2:
    #         continue
    #     images, dot1, dot2, good = seperate(plate)
    #     if good:
    #         break

    # if not good:
    #     return []
    # for image in images:
    #     if len(recognized_chars) == dot1 or len(recognized_chars) == dot2:
    #         recognized_chars.append('-')
    #     character = give_label_two_scores(image)
    #     if character != AMBIGUOUS_RESULT:
    #         recognized_chars.append(character)
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
        _, descriptor = sift.detectAndCompute(reference_characters[char], None)
        sift_database[char] = descriptor


def difference_score(test_image, reference_character):
    reference_character = cv2.resize(reference_character, (len(test_image[0]), len(test_image)))
    # return the number of non-zero pixels

    return np.count_nonzero(cv2.bitwise_xor(test_image, reference_character))


def give_label_two_scores(test_image):
    # Erode to remove noise
    test_image = cv2.erode(test_image, np.ones((2, 2)))

    difference_scores = []
    for key, value in reference_characters.items():
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

    ratio = A / B
    # print("ratio:", ratio)
    if ratio > 1 - EPSILON and ratio < 1 + EPSILON:
        ambiguous.append(A)
    #     return AMBIGUOUS_RESULT

    # Return a single character based on the lowest score
    return result_char, A


def write(plates):
    # open the file in the write mode
    with open('sampleOutput.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        header = ['License plate', 'Frame no.', 'Timestamp(seconds)']

        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)

        writer.writerow(['Total ambiguous', len(ambiguous)])


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


def get_horizontal_positions(plate):
    # use epsilon, this is
    epsilon = 0.05 * len(plate[0])
    # after observing multiple plates, we saw that each character has a width of approximately 10% of the plate's width
    char_width = 0.1 * len(plate[0])
    # boxes will contain 6 pairs of columns indices, called a box, where each pair is the horizontal interval of one character
    boxes = []
    # make sure boxes don't overlap
    overlap = np.array([])

    # find a pair six times, or stop when no space left 
    while (len(boxes) < 6 and len(overlap) < len(plate[0]) - int(char_width)):
        # minimum number of white pixels in column pair
        minwhite = 2 * len(plate)

        box = (0, 0)
        # for each column thats not in overlap
        for j in range(int(len(plate[0]) - char_width - epsilon)):
            if j not in overlap:
                # number of white pixels in first column
                whites = cv2.countNonZero(plate[:, j])
                # for the second column, try all indices between margin
                for jj in range(int(j + char_width), int(j + char_width + epsilon)):
                    # keep column pair that does not overlap other boxes and has the least amount of white pixels              
                    if (not np.in1d(overlap, np.arange(j, jj)).any()) and whites + cv2.countNonZero(
                            plate[:, jj]) < minwhite and cv2.countNonZero(plate[:, j:jj + 1]) > 5:
                        # update the minimum whites found
                        minwhite = whites + cv2.countNonZero(plate[:, jj])
                        # store pair
                        box = (j, jj)
        # update overlap
        overlap = np.concatenate((overlap, np.arange(box[0], box[1] - 2)))
        # store box
        boxes.append(box)

    # make sure all boxes are found, if not, replace boxes by simply dividing image by 6
    for box in boxes:
        if box[1] - box[0] < 2 or len(boxes) < 6:
            boxes = []
            for i in range(1, 7):
                boxes.append((int(i * (len(plate[0]) / 7.0) - (char_width / 2.0)),
                              int(i * (len(plate[0]) / 7.0) + (char_width / 2.0))))
            break

    boxes.sort()
    return boxes


def make_binary(image):
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
    return plate


def seperate(plate):
    # get horizontal character interval boundaries
    boxes = get_horizontal_positions(plate)

    # get positions of the two so called 'dots' in a license plate
    gaps = []
    for i in range(len(boxes) - 1):
        gaps.append(boxes[i + 1][0] - boxes[i][1])
    gaps = np.argsort(np.array(gaps))

    # two biggest gaps between characters
    dot1 = gaps[-1]
    dot2 = gaps[-2]
    # gap1 = int((boxes[dot1][1] + boxes[dot1 + 1][0]) / 2)
    # gap2 = int((boxes[dot2][1] + boxes[dot2 + 1][0]) / 2)

    # increase the second dot by one to make it correspond to an index later in the pipeline
    if dot1 > dot2:
        dot1 += 2
        dot2 += 1
    else:
        dot2 += 2
        dot1 += 1

    characters = []
    for box in boxes:
        char = plate[:, box[0]:box[1]]
        characters.append(char)
    return characters, dot1, dot2

    # # find height of two dots
    # # first manually find contours from the matching white pixels of the centers of the two just found gaps
    # black = True
    # matches = []
    # for i in range(len(plate)):
    #     if plate[i][gap1] == 255 and plate[i][gap2] == 255:
    #         if black:
    #             matches.append([i])
    #             black = False
    #     else:
    #         if not black:
    #             matches[-1].append(i)
    #             black = True

    # # now choose contour that is closests to the vertical center, and assign the middle of this contour, to finalindex
    # # we do this to find the vertical center of the plate's characters, since this is at the height of the two dots
    # finalindex = int(len(plate) / 2)
    # if len(matches) > 0:    
    #     # make sure the last contour also has a second value
    #     if len(matches[-1]) == 1:
    #         matches[-1].append(len(plate))
    #     # vertical center of plate
    #     center = float(len(plate) / 2)
    #     # minimum distance to keep track of and compare with
    #     mindis = len(plate)
    #     for m in matches:
    #         if np.abs(center - (float((m[1] + m[0]) / 2.0))) < mindis:
    #             # update minimum distance 
    #             mindis = np.abs(center - (float((m[1] + m[0]) / 2.0)))
    #             # update finalindex
    #             finalindex = int((m[1] + m[0]) / 2)

    # # after observing multiple plates, we saw that each character has a height of approximately 17% of the plate's width
    # heightchar = float(0.17 * len(plate[0]))
    # # if for some reason the finalindex is too high or too low for a character to fit, simply use the middle of the image's height
    # if finalindex < float(heightchar / 2.0) or finalindex > len(plate) - float(heightchar / 2.0):
    #     finalindex = int(len(plate) / 2)

    # finalindex += int(0.02 * len(plate))
    # # vertical boundaries
    # ymin = finalindex - int(heightchar / 2)
    # ymax = finalindex + int(heightchar / 2)

    # characters = []
    # for box in boxes:
    #     char = plate[ymin:ymax, box[0]:box[1]]

    #     # make sure no further errors will occur if character image is too small
    #     if len(char) < 2 or len(char[0]) < 2:
    #         return characters, dot1, dot2, False

    #     # crop character to boundary box
    #     char = crop_to_boundingbox(char)

    #     # again make sure no further error will occur if character image is too small
    #     if cv2.countNonZero(char) < 2 or len(char) < 2 or len(char[0]) < 2:
    #         return characters, dot1, dot2, False

    #     characters.append(char)

    # return characters, dot1, dot2, True
