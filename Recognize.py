import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np

# from Localization import get_plate

AMBIGUOUS_RESULT = "AMBIGUOUS"
EPSILON = 0.15
# Load the reference characters
character_array = ['B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                   '3', '4', '5', '6', '7', '8', '9']

reference_characters = {}
path = "TrainingSet/Categorie I/"
recognized_plates = []
scores = []
frames = []
same_car_plates = []
same_car_scores = []
sifts_numbers = {}
sifts_letters = {}
sift = cv2.xfeatures2d.SIFT_create(nfeatures=150)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)


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

# TODO check if 'is_digit' is necessary: no changes in score seen
def compare_neighbours(character_array, character_score, plate_score, is_digit):
    neighbours = {}
    # print(character_array)
    # print(character_score)
    # print(plate_score)
    for i, char in enumerate(character_array):
        if char.isdigit() != is_digit:
            continue

        weight = character_score[i] / plate_score[i][1] if plate_score[i][1] != 0 else 0  # plate_score[i][1]
        # print(weight)
        if char in neighbours:
            neighbours[char] += weight
        else:
            neighbours[char] = weight

    # print(neighbours)

    # return max(neighbours, key=neighbours.get)

    # TODO check if below is necessary: no changes in score seen

    best = max(neighbours, key=neighbours.get)
    chosen = [best]
    for key, value in neighbours.items():
        if key not in chosen and value == neighbours[best]:
            chosen.append(key)
    if len(chosen) == 1:
        return chosen[0]

    scores = {}

    for i in range(len(character_array)):
        if character_array[i] in chosen:
            scores[character_array[i]] = character_score[i]
    return max(scores, key=scores.get)


def segment_and_recognize(image, found, frame, compare):
    global same_car_plates, same_car_scores, frames

    # Call setup only once
    setup()
    # return immediately when nothing was not found in localization
    if not found:
        print('localization was wrong')
        return False

    # default dot positions
    dot1 = 2
    dot2 = 5

    append_known_plates = False
    plate_info = []
    recognized_chars = []

    # make sure no errors occur
    if len(image) > 1 and len(image[0]) > 1:
        binary = apply_isodata_thresholding(image)
        # binary = cv2.erode(binary, np.ones((2,2)))        
        # binary = cv2.dilate(binary, np.ones((2,2)))

        binary, found = remove_rows(binary)
        if not found:
            print("remove_rows found nothing")
            return False

        # make sure no errors occur
        if cv2.countNonZero(binary) != 0:
            # seperate characters from image
            char_images, dot1, dot2, found = contours(binary)
            if not found:
                char_images, dot1, dot2, found = segment(binary)
            if not found:
                return False
            # testsift(char_images)
            # recognize character images
            recognized_chars, score = get_recognized_chars(char_images, binary, dot1, dot2)
    # when the dots are at invalid positions found, or either the amount of characters found is not 6,
    # we assume it was no licence plate, so we try other contours
    if len(recognized_chars) != 6 or dot1 == 0 or dot2 == 0 or dot1 == 7 or dot2 == 7 or abs(dot1 - dot2) < 2 or abs(
            dot1 - dot2) > 4 or (dot1 > 3 and dot2 > 3) or (dot1 < 4 and dot2 < 4):
        print("invalid dot positions")
        return False

    # add dots ('-') at correct positions
    recognized = []
    scores_final = []
    for i in range(6):
        if len(recognized) == dot1 or len(recognized) == dot2:
            recognized.append('-')
            scores_final.append(0)
        recognized.append(recognized_chars[i])
        scores_final.append(score[i])
    if recognized.count('-') != 2:
        print("not 2 dots")
        return False





    plate = format_plate(recognized)
    new_plate = []
    final_frame = 0

    # plate_info.append(plate)
    # plate_info.append(frame)
    # plate_info.append(round(frame / 12))
    # recognized_plates.append(plate_info)
    # return True

    # if len(same_car_plates) > 1:
    #     unique, counts = np.unique(same_car_plates, return_counts=True)
    #     if max(counts) > 0.75 * len(same_car_plates):
    #         same_car_plates = [unique[np.argmax(counts)]]
    #         same_car_scores = [scores_final]
    #         compare = True

    if compare:
        print(same_car_plates)
        # Final frame of same plate, therefore time to compare
        # print("Same_car_scores:", same_car_scores)
        # print("Same_car_plates:", same_car_plates)

        if len(same_car_plates) == 0:
            # TODO check if this works
            # same_car_plates = [recognized]
            return False

        unique, counts = np.unique(same_car_plates, return_counts=True)
        best = max(counts)
        plates = []
        for i in range(len(counts)):
            if counts[i] == best:
                plates.append(unique[i])
        if len(plates) == 1:
            final_plate = plates[0]
        else:
            scores_per_plate = np.zeros(len(plates))
            for i in range(len(plates)):
                for j in range(len(same_car_plates)):
                    if same_car_plates[j] == plates[i]:
                        scores_per_plate[i] += sum(same_car_scores[j])
            final_plate = plates[np.argmin(scores_per_plate)]
        # if len(same_car_plates) > 1:
        #     is_digits1 = [0, 0]
        #     is_digits2 = [0, 0]
        #     is_digits3 = [0, 0]
        #     dash1_pos = np.zeros(8)
        #     dash2_pos = np.zeros(8)
        #     for plate in same_car_plates:
        #         dash1 = plate.index('-')
        #         dash2 = plate.index('-', dash1 + 1)
        #         if plate[dash1 - 1].isdigit():
        #             is_digits1[1] += 1
        #         else:
        #             is_digits1[0] += 1
        #         if plate[dash1 + 1].isdigit():
        #             is_digits2[1] += 1
        #         else:
        #             is_digits2[0] += 1
        #         if plate[dash2 + 1].isdigit():
        #             is_digits3[1] += 1
        #         else:
        #             is_digits3[0] += 1

        #         dash1_pos[dash1] += 1
        #         dash2_pos[dash2] += 1

        #     is_digits = []
        #     is_digits.append(False) if np.argmax(is_digits1) == 0 else is_digits.append(True)
        #     is_digits.append(False) if np.argmax(is_digits2) == 0 else is_digits.append(True)
        #     is_digits.append(False) if np.argmax(is_digits3) == 0 else is_digits.append(True)

        #     dashes = (np.argmax(dash1_pos), np.argmax(dash2_pos))

        #     if len(dashes) > 1:
        #         for i in range(len(same_car_plates)):
        #             dash1 = same_car_plates[i].index('-')
        #             dash2 = same_car_plates[i].index('-', dash1)
        #             if dash1 != dashes[0] or dash2 != dashes[1]:
        #                 same_car_plates[i] = same_car_plates[i].replace('-', '')
        #                 same_car_plates[i] = same_car_plates[i][:dashes[0]] + '-' + same_car_plates[i][
        #                                                                             dashes[0]:dashes[1] - 1] + '-' + \
        #                                      same_car_plates[i][dashes[1] - 1:]

        #     # Get score of full plate
        #     new_scores = []
        #     for i, score in enumerate(same_car_scores):
        #         new_scores.append((same_car_plates[i], sum(score)))

        #     # Compare plates to each other
        #     for i, char in enumerate(same_car_plates[-1]):
        #         if char == '-':
        #             new_plate.append('-')
        #             continue
        #         current_char = [char]
        #         current_score = [same_car_scores[-1][i]]
        #         # Loop over all characters in last found plate
        #         for j, current_plate in enumerate(same_car_plates):
        #             # Loop over all other plates of same car
        #             if current_plate is same_car_plates[-1]:
        #                 continue

        #             current_char.append(current_plate[i])
        #             current_score.append(same_car_scores[j][i])
        #         # print("current_char:", current_char)
        #         # print("current_score:", current_score)
        #         is_digit = is_digits[0] if i < dashes[0] else is_digits[1] if dashes[0] < i < dashes[1] else is_digits[2]

        #         new_scores = new_scores[-1:] + new_scores[:-1]
        #         # Create new plate using kNN implementation
        #         new_plate.append(compare_neighbours(current_char, current_score, new_scores, is_digit))  #

        # else:
        #     new_plate = same_car_plates[0]

        # # Create new plate using kNN implementation and format plate
        # final_plate = format_plate(new_plate)  # get_final_plate()
        # # print("final plate:", final_plate)

        # Update variables
        plate = final_plate
        append_known_plates = True
        same_car_plates = []
        same_car_scores = []
        final_frame = sum(frames) / len(frames) if len(frames) > 0 else 0
        frames = [frame]

    if not compare:
        # No comparison done yet, add plate
        same_car_plates.append(plate)
        same_car_scores.append(scores_final)
        frames.append(frame)

    # Append the recognised characters, frame no. and time (rounded down)
    plate_info.append(plate)
    plate_info.append(final_frame)
    plate_info.append(round(final_frame / 12))

    # Add to list of known plates, only if final plate is known
    if append_known_plates:
        recognized_plates.append(plate_info)
    scores.append(scores_final)

    return compare


def format_plate(plate):
    final_plate = ''
    for char in plate:
        final_plate += char
    return final_plate


def get_recognized_chars(images, plate, dot1, dot2):
    testbram = []
    final = (0, len(plate) - 1)
    final_characters = []
    final_scores = [float('inf')] * 6

    # we assumed each character has a height of approximately 17% of the plates width
    char_height = int(float(0.18 * len(plate[0])))
    margin = int(0.1 * char_height)

    # try for all heights and choose the one where all the characters combined have the best diff_score
    for height in range(char_height - margin, char_height + margin + 1):
        for i in range(max(1, len(plate) - height)):
            start_index = i
            end_index = min(i + height, len(plate))
            recognized_chars, character_scores = recognize_characters(images, dot1, dot2, start_index, end_index)
            # check if best score and update variables if needed
            if sum(character_scores) < sum(final_scores):
                final = (start_index, end_index)
                final_characters = recognized_chars
                final_scores = character_scores
    # returns array of all the (hopefully 6) recognized chars
    return final_characters, final_scores


def recognize_characters(images, dot1, dot2, start_index, end_index):
    chars = []
    char_scores = []
    split_points = [0, dot1, dot2 - 1, len(images)]
    for j in range(0, 3):
        numbers = []
        letters = []
        num_scores = []
        let_scores = []
        for jj in range(len(images)):
            if jj >= split_points[j] and jj < split_points[j + 1]:
                image = images[jj]
                # crop image to certain height
                cropped = image[start_index:end_index]
                cropped = cv2.erode(cropped, np.ones((3, 3)))
                cropped = cv2.dilate(cropped, np.ones((3, 3)))
                # crop image to bounding box
                bounding = cv2.boundingRect(cropped)
                cropped = cropped[bounding[1]:bounding[1]+bounding[3],bounding[0]:bounding[0]+bounding[2]]
                # make sure no errors occur
                if len(cropped) < 2 or len(cropped[0]) < 2:
                    final_scores = [float('inf')]
                    break

                # get the character with the best score
                number, score1 = give_label_two_scores(cropped, True)
                letter, score2 = give_label_two_scores(cropped, False)
                numbers.append(number)
                num_scores.append(score1)
                letters.append(letter)
                let_scores.append(score2)
        if sum(num_scores) < sum(let_scores):
            char_scores += num_scores
            chars += numbers
        else:
            char_scores += let_scores
            chars += letters
    return chars, char_scores


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
        bounding = cv2.boundingRect(value)
        img = value[bounding[1]:bounding[1]+bounding[3],bounding[0]:bounding[0]+bounding[2]]
        reference_characters[char] = img
        # img = cv2.resize(img, (100,100))
        desc = sift_descriptor(img)
        if char.isdigit():
            sifts_numbers[char] = desc
        else:
            sifts_letters[char] = desc


def difference_score(test_image, reference_character):
    reference_character = cv2.resize(reference_character, (len(test_image[0]), len(test_image)))
    # return the number of non-zero pixels
    return np.count_nonzero(cv2.bitwise_xor(test_image, reference_character))


# get gradient method used in the lab
def get_gradient(image):
    # Sobel gradient in x and y direction
    Sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    I_x = cv2.filter2D(np.float64(image), -1, Sobel_kernel_x)
    I_y = cv2.filter2D(np.float64(image), -1, Sobel_kernel_y)
    # Gradient magnitude
    gradient = np.hypot(I_x, I_y)
    # Gradient orientation
    I_x[I_x == 0] = 0.0001
    theta = np.arctan(I_y / I_x)
    return gradient, theta


# sift descriptor used in the lab
def sift_descriptor(image):
    image = cv2.resize(image, (30, 20))
    N = 3
    result = []
    # Take only 16x16 window of the picture from the center
    iboundaries = np.arange(len(image) + 1, step=len(image) / N).astype(int)
    jboundaries = np.arange(len(image[0]) + 1, step=len(image[0]) / N).astype(int)
    for i in range(N):
        for j in range(N):
            subwindow = image[iboundaries[i]:iboundaries[i + 1], jboundaries[j]:jboundaries[j + 1]]
            mag, ang = get_gradient(subwindow)

            hist, bin_edges = np.histogram(ang, bins=8, weights=mag)
            for value in hist:
                result.append(value)
            # Add the direction of the edge to the feature vector, scaled by its magnitude
    result = np.array(result)
    result /= np.linalg.norm(result)
    return result


def give_label_two_scores_sift(image, is_digit):
    differences = {}
    desc = sift_descriptor(image)

    if is_digit:
        sifts = sifts_numbers
    else:
        sifts = sifts_letters

    for key in sifts:
        differences[key] = np.linalg.norm(desc - sifts[key])

    char = min(differences, key=differences.get)
    # plotImage(image, str(char))
    return char, differences[char]


def give_label_two_scores(test_image, is_digit):
    # Erode to remove noise
    test_image = cv2.erode(test_image, np.ones((2, 2)))

    # get all difference scores
    difference_scores = []
    for key, value in reference_characters.items():
        if key.isdigit() == is_digit:
            difference_scores.append(difference_score(test_image, value))
        else:
            difference_scores.append(float('inf'))

    # get two lowest scores
    difference_scores = np.array(difference_scores)
    sorted_indices = np.argsort(difference_scores)
    A = difference_scores[sorted_indices[0]]
    B = difference_scores[sorted_indices[2]]

    # get characters corresponding to these two lowest scores
    result_char_1 = list(reference_characters)[sorted_indices[0]]
    result_char_2 = list(reference_characters)[sorted_indices[1]]

    # if ambiguous, choose one with best score using sift descriptor
    if B == 0 or (A / B > 1 - EPSILON and A / B < 1 + EPSILON):
        our_sift = sift_descriptor(test_image)
        sift_1 = sift_descriptor(reference_characters[result_char_1])
        sift_2 = sift_descriptor(reference_characters[result_char_2])
        diff_1 = np.linalg.norm(sift_1 - our_sift)
        diff_2 = np.linalg.norm(sift_2 - our_sift)
        if diff_1 > diff_2:
            return result_char_2, A

    return result_char_1, A


def write(plates):
    # open the file in the write mode
    with open('Output.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        header = ['License plate', 'Frame no.', 'Timestamp(seconds)']

        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)


def crop_width(image):
    found = False
    for j in range(len(image[0])):
        for i in range(len(image)):
            if image[i][j] != 0:
                start = j
                found = True
                break
        if found:
            break

    for j in range(1, len(image[0]) + 1):
        for i in range(len(image)):
            if image[i][-j] != 0:
                end = j
                return image[:, start:len(image[0]) - end + 1]
    return image



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


def apply_thresholding(image):
    # convert to grayscale
    plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Make use of Otsu Thresholding to make plate binary image
    ret, thresh = cv2.threshold(plate, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure that characters are white and background is black
    return 255 - thresh


def apply_isodata_thresholding(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    epsilon = 0.005
    # Compute the histogram and set up variables
    hist = np.array(cv2.calcHist([image], [0], None, [256], [0, 256])).flatten()
    t = np.mean(image)
    old_t = -2 * epsilon

    # Iterations of the isodata thresholding algorithm
    while (abs(t - old_t) >= epsilon):
        sum1 = 0
        sum2 = 0
        for i in range(0, int(t)):
            sum1 = sum1 + hist[i] * i
            sum2 = sum2 + hist[i]
        m1 = sum1 / sum2
        sum1 = 0
        sum2 = 0
        for i in range(int(t), len(hist)):
            sum1 = sum1 + i * hist[i]
            sum2 = sum2 + hist[i]
        m2 = sum1 / sum2
        # TODO Calculate new tau
        old_t = t
        t = (m1 + m2) / 2
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > t:
                image[i][j] = 0
            else:
                image[i][j] = 255
    # image = cv2.erode(image, np.ones((1, 5)))
    # image = cv2.dilate(image, np.ones((1,5)))
    return image


def remove_rows(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    invalid_rows = []
    good_rows = []
    count = 0
    for i in range(len(image)):
        row = image[i]
        white = False
        count_whites = 0
        count_blacks = 0
        continues_whites = 0
        continues_blacks = 0
        for pixel in row:
            if white:
                if pixel == 0:
                    white = False
                    continues_whites = max(continues_whites, count_whites)
                    count_whites = 0
                    count_blacks = 1
                else:
                    count_whites += 1
            else:
                if pixel != 0:
                    white = True
                    continues_blacks = max(continues_blacks, count_blacks)
                    count_blacks = 0
                    count_whites = 1
                else:
                    count_blacks += 1
        continues_whites = max(continues_whites, count_whites)
        continues_blacks = max(continues_blacks, count_blacks)

        whites = cv2.countNonZero(row)
        if whites > 0.7 * len(row) or whites < 0.1 * len(row):
            invalid_rows.append(i)
    if len(invalid_rows) == 0:
        return [], False

    crop = (0, invalid_rows[0])
    for i in range(1, len(invalid_rows)):
        if invalid_rows[i] - invalid_rows[i - 1] > crop[1] - crop[0]:
            crop = (invalid_rows[i - 1], invalid_rows[i])

    return image[crop[0]:crop[1] + 1], True
    #     if whites < 0.7 * len(row) and whites > 0.1 * len(row) and continues_blacks < 0.3 * len(
    #             row) and continues_whites < 0.2 * len(row):
    #         good_rows.append(i)

    # char_height = int(float(0.16 * len(image[0])))

    # if len(good_rows) < 20:
    #     return [], False

    # start_crop = 0
    # end_crop = len(image) - 1

    # if 1 + good_rows[-1] - good_rows[0] <= char_height:
    #     # start_crop = max(0, good_rows[-1] - char_height + 1)
    #     # end_crop = min(len(image), good_rows[0] + char_height)

    #     start_crop = max(0, int(good_rows[0] - 0.5 * (char_height - (good_rows[-1] - good_rows[0]))))
    #     end_crop = min(len(image), int(good_rows[-1] + 0.5 * (char_height - (good_rows[-1] - good_rows[0]))))
    # else:
    #     interval = (0, len(image) - 1)
    #     good_rows = np.array(good_rows)
    #     best = 0
    #     for i in range(good_rows[0], good_rows[-1] + 1):
    #         chosen_rows = good_rows[((i <= good_rows) & (good_rows < i + char_height))]
    #         if len(chosen_rows) > best:
    #             best = len(chosen_rows)
    #             interval = (chosen_rows[0], chosen_rows[-1])
    #         elif len(chosen_rows) == best:
    #             interval = (interval[0], chosen_rows[-1])

    #     start_crop = max(0, interval[1] - char_height + 1)
    #     end_crop = min(len(image), interval[0] + char_height)

    # image = image[start_crop:end_crop]
    # newimage = []
    # for row in image:
    #     whites = cv2.countNonZero(row)
    #     if whites < 0.8 * len(image[0]) and whites > 0.2 * len(image[0]):
    #         newimage.append(row)
    # image = np.array(newimage)
    # return image, True


def separate(plate):
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

    # increase the second dot by one to make it correspond to an index later in the pipeline
    if dot1 > dot2:
        temp = dot1
        dot1 = dot2
        dot2 = temp
    dot1 += 1
    dot2 += 2

    # return all characters
    characters = []
    for box in boxes:
        char = plate[:, box[0]:box[1]]
        characters.append(char)
    return characters, dot1, dot2


def segment(image):
    # eroded = cv2.erode(image, np.ones((3,3)))
    # eroded = cv2.dilate(eroded, np.ones((3,3)))
    char_width = 0.1 * len(image[0])
    margin = 0.4 * char_width

    whites_per_column = []
    for j in range(len(image[0])):
        whites = cv2.countNonZero(image[:, j])
        while len(whites_per_column) <= whites:
            whites_per_column.append([])
        whites_per_column[whites].append(j)

    boxes = []
    columns_below_threshold = []
    for columns in whites_per_column:
        if len(columns) < 1:
            continue

        if len(columns_below_threshold) == 0:
            columns_below_threshold = columns
        else:
            columns_below_threshold = sorted(columns_below_threshold + columns)

        index_tuples = []
        tuple_widths = []
        continue_loop = False
        for i in range(len(columns_below_threshold) - 1):
            # bgr[:, columns_below_threshold[i]] = [0, 255, 0]
            width = columns_below_threshold[i + 1] - columns_below_threshold[i]
            if width > char_width + margin:
                continue_loop = True
                break
            else:
                index_tuples.append((columns_below_threshold[i], columns_below_threshold[i + 1]))
                tuple_widths.append(width)

        if continue_loop:
            continue

        if len(index_tuples) >= 6:
            boxes = []
            indices = np.argsort(tuple_widths)
            indices = indices[len(indices) - 6:]
            indices = np.sort(indices)
            for i in indices:
                boxes.append(index_tuples[i])
            break

    if len(boxes) != 6:
        return [], 1, 1, False

    character_images = []
    gaps = []
    for i in range(len(boxes)):
        character_images.append(image[:, boxes[i][0]:boxes[i][1]])
        if i != 0:
            gaps.append(boxes[i][0] - boxes[i - 1][1])
    gap_indices = np.argsort(gaps)
    dash1 = min(gap_indices[-1], gap_indices[-2]) + 1
    dash2 = max(gap_indices[-1], gap_indices[-2]) + 2

    # bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for box in boxes:
    #     bgr[:2,box[0]:box[1]] = [0, 255, 0]
    # print(boxes)
    # plotImage(bgr, str(dash1)+", "+str(dash2))

    return character_images, dash1, dash2, True

    # boxes = []
    # in_box = False
    # for j in range(len(image[0])):
    #     if cv2.countNonZero(eroded[:,j]) > 0.1*len(image):
    #         if not in_box:
    #             in_box = True
    #             boxes.append((j,j))
    #     else:
    #         if in_box:
    #             in_box = False
    #             boxes[-1] = (boxes[-1][0], j)
    # if in_box:
    #     boxes[-1] = (boxes[-1][0], len(image[0])-1)

    # boxes_width = []
    # for box in boxes:
    #     boxes_width.append(box[1]-box[0])
    # boxes_width = np.argsort(boxes_width)
    # if len(boxes) < 6:
    #     print("probleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeem")
    #     return [], 0, 0, False 

    # boxes_width = boxes_width[len(boxes_width)-6:]

    # gaps = []
    # boxes_final = []
    # characters = []
    # boxes_width = np.sort(boxes_width)
    # for i in boxes_width:
    #     boxes_final.append(boxes[i])
    #     characters.append(image[:,boxes[i][0]:boxes[i][1]])

    # for i in range(len(boxes_final)-1):
    #     gaps.append(boxes_final[i+1][0]-boxes_final[i][1])
    # gaps = np.argsort(gaps)
    # dash1 = min(gaps[-1],gaps[-2])
    # dash2 = max(gaps[-1],gaps[-2])

    # dash1 += 1
    # dash2 += 2

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for box in boxes_final:
    #     image[:,box[0]] = [0, 255, 0]
    #     image[:,box[1]] = [0, 255, 0]
    # plotImage(image, str(dash1)+", "+str(dash2))

    # return characters, dash1, dash2, True


def contours(image):
    image = cv2.erode(image, np.ones((1,1)))
    image = cv2.dilate(image, np.ones((1,1)))
    char_height = float(0.16 * len(image[0]))
    char_width = 0.1 * len(image[0])
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = []
    indices = []
    areas = []
    for i in range(len(contours)):
        contour = contours[i]
        # store the pixels of current contour
        box = cv2.boundingRect(contour)
        # print(boundingBox)
        # plotImage(image[box[1]:box[1]+box[3],box[0]:box[0]+box[2]])
        if box[2] > 0.7 * char_width and box[2] < 1.3 * char_width and box[3] > 0.7 * char_height and box[
            3] < 1.3 * char_height:
            # print(box)
            # pixels = np.zeros((len(image), len(image[0])))
            # for pixel in contour:
            #     i = pixel[0][1]
            #     j = pixel[0][0]
            #     pixels[i][j] = 255
            # plotImage(pixels)
            bounding_boxes.append(box)
            indices.append(i)

    without_child_contours = []
    for i in range(len(bounding_boxes)):
        index = indices[i]
        if hierarchy[0][index][3] not in indices:
            without_child_contours.append(bounding_boxes[i])
        # else:
        #     indices = np.delete(hierarchy[i][3])
    bounding_boxes = without_child_contours

    if len(bounding_boxes) < 6:
        return [], 0, 0, False

    if len(bounding_boxes) > 6:

        print("meer dan 6")
        ratios = []
        for box in bounding_boxes:
            ratios.append(float(box[3] / box[2]))
        print(bounding_boxes)
        print(ratios)
        indices_sorted = np.argsort(ratios)
        six_most_alike = []
        min_diff = float('inf')
        for i in range(len(indices_sorted) - 5):
            diff = np.abs(ratios[indices_sorted[i]] - ratios[indices_sorted[i + 1]])
            diff += np.abs(ratios[indices_sorted[i + 1]] - ratios[indices_sorted[i + 2]])
            diff += np.abs(ratios[indices_sorted[i + 2]] - ratios[indices_sorted[i + 3]])
            diff += np.abs(ratios[indices_sorted[i + 3]] - ratios[indices_sorted[i + 4]])
            diff += np.abs(ratios[indices_sorted[i + 4]] - ratios[indices_sorted[i + 5]])
            if diff < min_diff:
                min_diff = diff
                six_most_alike = [bounding_boxes[i], bounding_boxes[i + 1], bounding_boxes[i + 2],
                                  bounding_boxes[i + 3], bounding_boxes[i + 4], bounding_boxes[i + 5]]
        print(six_most_alike)
        bounding_boxes = six_most_alike

    bounding_boxes = sorted(bounding_boxes)
    character_images = []
    gaps = []
    for i in range(len(bounding_boxes)):
        character_images.append(image[bounding_boxes[i][1]:bounding_boxes[i][1] + bounding_boxes[i][3],
                                bounding_boxes[i][0]:bounding_boxes[i][0] + bounding_boxes[i][2]])
        if i != 0:
            gaps.append(bounding_boxes[i][0] - bounding_boxes[i - 1][0] + bounding_boxes[i - 1][2])
    gap_indices = np.argsort(gaps)
    dash1 = min(gap_indices[-1], gap_indices[-2]) + 1
    dash2 = max(gap_indices[-1], gap_indices[-2]) + 2

    return character_images, dash1, dash2, True
