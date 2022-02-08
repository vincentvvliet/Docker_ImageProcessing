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


def segment_and_recognize(image, found, frame):
    # Call setup only once
    setup()

    plate_info = []
    # return immediately when nothing was not found in localization
    if not found:
        return
    # default dot positions
    dot1 = 2
    dot2 = 5

    recognized_chars = []
    # make sure no errors occur
    if len(image) > 1 and len(image[0]) > 1:
        binary = apply_thresholding(image)        
        # binary = cv2.dilate(binary, np.ones((4,4)))

        # binary = cv2.erode(binary, np.ones((4,4)))
        binary, found = remove_rows(binary)
        if not found:
            return
        # make sure no errors occur
        if cv2.countNonZero(binary) != 0:
            # seperate characters from image
            char_images, dot1, dot2 = separate(binary)
            # recognize character images
            recognized_chars, score = get_recognized_chars(char_images, binary, dot1, dot2)

    # when the dots are at invalid postions found, or either the amount of characters found is not 6, 
    # we assume it was no licence plate, so we try other contours
    if len(recognized_chars) != 6 or dot1 == 0 or dot2 == 0 or dot1 == 7 or dot2 == 7 or abs(dot1 - dot2) < 2 or abs(
            dot1 - dot2) > 4 or (dot1 > 3 and dot2 > 3) or (dot1 < 4 and dot2 < 4):
        return

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
        return

    plate = ''
    for char in recognized:
        plate += char

    # Append the reconised characters, frame no. and time (rounded down)
    plate_info.append(plate)
    plate_info.append(frame)
    plate_info.append(round(frame / 12))

    # Add to list of known plates
    recognized_plates.append(plate_info)
    scores.append(scores_final)

    # Only write at the end
    if frame > 2000:
        write(recognized_plates)

def get_recognized_chars(images, plate, dot1, dot2):
    # testbram = []
    final = (0,len(plate)-1)
    final_characters = []      
    final_scores = [float('inf')] * 6

    # we assumed each character has a height of approximately 17% of the plates width
    char_height = int(float(0.17 * len(plate[0])))
    margin = int(0.1*char_height)
    margin = 0

    if(len(plate) <= char_height+margin):
        char_height = len(plate) - margin - 1

    # try for all heights and choose the one where all the characters combined have the best diff_score
    for start_index in range(len(plate)-char_height):
        for end_index in range(start_index+char_height, min(len(plate),start_index+char_height+margin+1)):

            recognized_chars, character_scores = recognize_characters(images, dot1, dot2, start_index, end_index)

            # check if best score and update variables if needed
            if sum(character_scores) < sum(final_scores):
                final = (start_index, end_index)
                final_characters = recognized_chars
                final_scores = character_scores
                # testbram = testbram1

    # returns array of all the (hopefully 6) recognized chars
    return final_characters, final_scores

def recognize_characters(images, dot1, dot2, start_index, end_index):
    # testbram1 = []
    chars = []
    char_scores = []
    split_points = [0, dot1, dot2-1, len(images)]
    for j in range(0, 3):
        numbers = []
        letters = []
        num_scores = []
        let_scores = []
        for jj in range(len(images)):
            if jj >= split_points[j] and jj < split_points[j+1]:
                image = images[jj]
                # crop image to certain height
                cropped = image[start_index:end_index]
                cropped = cv2.erode(cropped, np.ones((3,3)))
                cropped = cv2.dilate(cropped, np.ones((3,3)))
                # crop image to bounding box
                cropped = crop_to_boundingbox(cropped)
                # testbram1.append(cropped)
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
        reference_characters[char] = crop_to_boundingbox(value)


            

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
    image = cv2.resize(image, (16, 16))
    result = []
    # Take only 16x16 window of the picture from the center
    boundaries = [0, 4, 8, 12]
    for i in boundaries:
        for j in boundaries:
            subwindow = image[i:i + 4, j:j + 4]
            mag, ang = get_gradient(subwindow)
            hist, bin_edges = np.histogram(ang, bins=8, weights=mag)
            for value in hist:
                result.append(value)
    result = np.array(result)
    result /= np.linalg.norm(result)
    return result



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


def crop_to_boundingbox(image):
    if len(image) < 2 or len(image[0]) < 2:
        return image
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
    old_t = -2*epsilon
    
    # Iterations of the isodata thresholding algorithm
    while(abs(t - old_t) >= epsilon):
        sum1 = 0
        sum2 = 0
        for i in range(0,int(t)):
            sum1 = sum1 + hist[i] * i
            sum2 = sum2 + hist[i]
        m1 = sum1 / sum2
        sum1 = 0
        sum2 = 0
        for i in range(int(t),len(hist)):
            sum1 = sum1 + i * hist[i]
            sum2 = sum2 + hist[i]
        m2 = sum1 / sum2
        #TODO Calculate new tau
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
        if whites < 0.7*len(row) and whites > 0.1*len(row) and continues_blacks < 0.3*len(row) and continues_whites < 0.2*len(row):
            good_rows.append(i)

    char_height = int(float(0.17 * len(image[0])))

    if len(good_rows) < 20:
        return [], False

    start_crop = 0
    end_crop = len(image)-1
    first_good_row = good_rows[0]
    last_good_row = good_rows[-1]
    if 1 + last_good_row-first_good_row <= char_height:
        start_crop = max(0, last_good_row-char_height)
        end_crop = min(len(image)-1, first_good_row+char_height)
    else:
        good_rows = np.array(good_rows)
        best = 0
        for i in range(first_good_row, last_good_row+1):
            j = ((i <= good_rows) & (good_rows <= i + char_height)).sum()
            if j > best:
                best = j
                start_crop = i
                end_crop = min(len(image)-1, i + char_height)
            elif j == best:
                end_crop = min(len(image)-1, i + char_height)
    image = image[start_crop:end_crop+1]

    newimage = []
    for row in image:
        whites = cv2.countNonZero(row)
        if whites < 0.8*len(image[0]) and whites > 0.2*len(image[0]):
            newimage.append(row)
    image = np.array(newimage)
    # img = []
    # for i in range(len(image)):
    #     row = []
    #     for p in image[i]:
    #         if i in good_rows:
    #             if p == 0:
    #                 row.append([0,0,0])
    #             else:
    #                 row.append([255,255,255])
    #         else:
    #             row.append([0, 255, 0])
    #     img.append(np.array(row))
    # img = np.array(img)
    return image, True
    # img[start_crop, :] = [255, 255, 255]
    # img[end_crop, :] = [255, 255, 255]
    # plotImage(img)
def separate(plate):
    # get horizontal character interval boundaries
    boxes = get_horizontal_positions(plate)

    # current_image = []
    # for row in plate:
    #     roww = []
    #     for p in row:
    #         if p == 0:
    #             roww.append([0,0,0])
    #         else:
    #             roww.append([255,255,255])
    #     current_image.append(np.array(roww))
    # current_image = np.array(current_image)

    # for box in boxes:
    #     # current_image[0:1,box[0]:box[1]+1] = [0, 255, 0]
    #     current_image[:,box[0]] = [0, 255, 0]
    #     current_image[:, box[1]] = [0, 255, 0]
    # plotImage(current_image)

    # invalid_rows = []
    # count = 0
    # for i in range(len(plate)):
    #     found = False
    #     for j in range(len(plate[i])):
    #         for box in boxes:
    #             if plate[i][j] == 255 and (box[0] == j or box[1] == j):
    #                 invalid_rows.append(min(i, len(plate)-1-i))
    #                 count += 1
    #                 found = True
    #                 break
    #         if found:
    #             break
    #     if not found:
    #         invalid_rows.append(float('inf'))

    # char_height = int(float(0.17 * len(plate[0])))
    # invalid_rows = np.argsort(invalid_rows)
    # invalid_rows = invalid_rows[:min(len(plate)-char_height, count+1)]
    # invalid_rows = -np.sort(-invalid_rows)
    # for i in invalid_rows:
    #     plate = np.delete(plate, i, axis=0)

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
