import cv2
import matplotlib.pyplot as plt
import numpy as np

AMBIGUOUS_RESULT = "AMBIGUOUS"
EPSILON = 0.15
# Load the reference characters
character_array = ['B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                   '3', '4', '5', '6', '7', '8', '9']

reference_characters = {}
same_car_plates = []
same_car_scores = []
sifts_numbers = {}
sifts_letters = {}


def plotImage(img, title=""):
    """Image plotting for debugging."""
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def loadImage(filepath, filename, grayscale=True):
    """Load image from path."""
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


def segment_and_recognize(image, found):
    """ Main functionality for pipeline.

    Calls setup function.
    Applies thresholding and removes unnecessary rows of pixels.
    Finds the separate images via existing contours. If this fails, tries the same by segmenting the characters.
    Checks whether all went well and then formats the plate to add to the result data structure.
    """

    # return immediately when nothing was not found in localization
    if not found:
        # Localization failed, plate will be invalid. Therefore, return.
        return '', 0

    # default dot positions
    dot1 = 2
    dot2 = 5
    score = 0

    recognized_chars = []

    # make sure no errors occur
    if len(image) > 1 and len(image[0]) > 1:
        binary = apply_isodata_thresholding(image)
        binary = cv2.erode(binary, np.ones((2, 2)))
        binary = cv2.dilate(binary, np.ones((2, 2)))

        binary, found = crop_height(binary)
        if not found:
            # Remove rows function detected invalid state
            return '', 0

        # make sure no errors occur
        if cv2.countNonZero(binary) != 0:
            # seperate characters from image
            char_images, dot1, dot2, found = segment_via_contours(binary)
            if not found:
                char_images, dot1, dot2, found = segmentation_alternative(binary)
                if not found:
                    return '', 0
                # recognize character images
                recognized_chars, score = get_recognized_chars(char_images, binary, dot1, dot2)
            else:
                # recognize character images
                recognized_chars, score = get_from_contours(char_images, dot1, dot2)

    # when the dots are at invalid positions found, or either the amount of characters found is not 6,
    # we assume it was no licence plate, so we try other contours
    if len(recognized_chars) != 6 or dot1 == 0 or dot2 == 0 or dot1 == 7 or dot2 == 7 or abs(dot1 - dot2) < 2 or abs(
            dot1 - dot2) > 4 or (dot1 > 3 and dot2 > 3) or (dot1 < 4 and dot2 < 4):
        return '', 0

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
        # Not the correct amount of dashes found, therefore invalid plate
        return '', 0

    return format_plate(recognized), sum(scores_final)


def setup():
    """ Setup function to be called once at the start of the pipeline. """

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
        [X, Y, W, H] = cv2.boundingRect(value)
        img = value[Y:Y + H, X:X + W]
        reference_characters[char] = img
        desc = sift_descriptor(img)
        if char.isdigit():
            sifts_numbers[char] = desc
        else:
            sifts_letters[char] = desc


def apply_isodata_thresholding(image):
    """Apply thresholding using the ISODATA method."""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    epsilon = 0.005
    # Compute the histogram and set up variables
    hist = calculateHistogram(image)
    t = np.mean(image)
    old_t = -2 * epsilon

    # Iterations of the ISODATA thresholding algorithm
    while abs(t - old_t) >= epsilon:
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
        old_t = t
        t = (m1 + m2) / 2
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > t:
                image[i][j] = 0
            else:
                image[i][j] = 255

    return image


def calculateHistogram(image):
    """Calculate histogram for input greyscale image."""

    hist = np.zeros(256)
    for row in image:
        for pixel in row:
            hist[pixel] += 1

    return hist


def crop_height(image):
    """Removes rows that are more than 70% or less than 10% white as they contain no relevant information."""

    invalid_rows = []
    for i in range(len(image)):
        row = image[i]
        white = False
        count_whites = 0
        count_blacks = 0
        continuous_whites = 0
        continuous_blacks = 0
        # find all rows that most likely don't overlap with a character
        for pixel in row:
            # calculate the biggest amount of continues white pixels in this row, and do the same for black pixels
            if white:
                if pixel == 0:
                    white = False
                    continuous_whites = max(continuous_whites, count_whites)
                    count_whites = 0
                    count_blacks = 1
                else:
                    count_whites += 1
            else:
                if pixel != 0:
                    white = True
                    continuous_blacks = max(continuous_blacks, count_blacks)
                    count_blacks = 0
                    count_whites = 1
                else:
                    count_blacks += 1

        # amount of white pixels in the row
        whites = cv2.countNonZero(row)
        # if row contains eiter too much white, or too much black, store it as an invalid row
        if whites > 0.7 * len(row) or whites < 0.1 * len(row):
            invalid_rows.append(i)
    # if no invalid rows found, just return image
    if len(invalid_rows) == 0:
        return image, True

    # choose two invalid rows that have the biggest distance between them, and make those rows the bounding positions
    crop = (0, invalid_rows[0])
    for i in range(1, len(invalid_rows)):
        if invalid_rows[i] - invalid_rows[i - 1] > crop[1] - crop[0]:
            crop = (invalid_rows[i - 1], invalid_rows[i])

    return image[crop[0]:crop[1] + 1], True


def segment_via_contours(image):
    """Segment by making use of the contours."""

    # Morphological operations to remove noise
    image = cv2.erode(image, np.ones((1, 1)))
    image = cv2.dilate(image, np.ones((1, 1)))

    # approximate the height of a character
    char_height = float(0.16 * len(image[0]))
    # approximate the height of a character
    char_width = 0.1 * len(image[0])
    # get all contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = []
    indices = []
    # for all contours store bounding boxes with satisfying size
    for i, contour in enumerate(contours):
        # store the pixels of current contour
        box = cv2.boundingRect(contour)
        # if bounding box of a contour has a width and height equal to char_height and char_width with margin 30%, we store it
        if 0.7 * char_width < box[2] < 1.3 * char_width and 0.7 * char_height < box[3] < 1.3 * char_height:
            bounding_boxes.append(box)
            indices.append(i)

    # make sure we only use contours that is not a child of any other chosen contour
    without_child_contours = []
    for i, bounding_box in enumerate(bounding_boxes):
        if hierarchy[0][indices[i]][3] not in indices:
            without_child_contours.append(bounding_box)
    bounding_boxes = without_child_contours

    # if less than 6 bounding boxes found, return false
    if len(bounding_boxes) < 6:
        return [], 0, 0, False

    # if more than 6 bounding boxes, choose six most alike in terms of width/height ratio
    if len(bounding_boxes) > 6:
        bounding_boxes = find_most_alike(bounding_boxes)

    # we now have exactly six bounding boxes, sort the list to remain same oder as in the image
    bounding_boxes = sorted(bounding_boxes)
    character_images = []
    gaps = []
    # loop through all six boxes
    for i, bounding_box in enumerate(bounding_boxes):
        character_images.append(image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                                bounding_box[0]:bounding_box[0] + bounding_box[2]])
        if i != 0:
            # store the distances between the boxes
            gaps.append(bounding_boxes[i][0] - bounding_boxes[i - 1][0] + bounding_boxes[i - 1][2])

    # choose two biggest gaps as dash positions and convert to correct index
    gap_indices = np.argsort(gaps)
    dash1 = min(gap_indices[-1], gap_indices[-2]) + 1
    dash2 = max(gap_indices[-1], gap_indices[-2]) + 2

    return character_images, dash1, dash2, True


def find_most_alike(bounding_boxes):
    """Find bounding boxes that are most alike in width:height ratio."""

    # Store all ratios
    ratios = []
    for box in bounding_boxes:
        ratios.append(float(box[3] / box[2]))
    # get sorted indices of ratios
    indices_sorted = np.argsort(ratios)
    six_most_alike = []
    min_diff = float('inf')
    # loop through the ratios, using the sorted indices s.t we loop through sorted ratios in descending order
    for i in range(len(indices_sorted) - 5):
        # for each uninterrupted combination of six ratios, calculate the total difference between them
        diff = np.abs(ratios[indices_sorted[i]] - ratios[indices_sorted[i + 1]])
        diff += np.abs(ratios[indices_sorted[i + 1]] - ratios[indices_sorted[i + 2]])
        diff += np.abs(ratios[indices_sorted[i + 2]] - ratios[indices_sorted[i + 3]])
        diff += np.abs(ratios[indices_sorted[i + 3]] - ratios[indices_sorted[i + 4]])
        diff += np.abs(ratios[indices_sorted[i + 4]] - ratios[indices_sorted[i + 5]])
        # keep the six ratios with lowest difference
        if diff < min_diff:
            # update variables
            min_diff = diff
            six_most_alike = [bounding_boxes[i], bounding_boxes[i + 1], bounding_boxes[i + 2],
                              bounding_boxes[i + 3], bounding_boxes[i + 4], bounding_boxes[i + 5]]
    # update the bounding_boxes
    bounding_boxes = six_most_alike

    return bounding_boxes


def segmentation_alternative(image):
    """Alternative method to segment characters only if segmentation via contours failed."""

    # determine char_width with margin
    char_width = 0.1 * len(image[0])
    margin = 0.4 * char_width

    # for each column, map the amount of white pixels to the column index, in such a way that whites_per_column[x]
    # contains all the columns that have exactly x white pixels
    whites_per_column = []
    for j in range(len(image[0])):
        whites = cv2.countNonZero(image[:, j])
        while len(whites_per_column) <= whites:
            whites_per_column.append([])
        whites_per_column[whites].append(j)

    boxes = []
    # for each column in whites_per_column, also map it to a lower amount of whites
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

        # now try each index as threshold for the amount of whites per column, until we found the threshold that segments our characters the best
        for i in range(len(columns_below_threshold) - 1):
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

    # find two biggest gaps between boxes to determine dash positions
    character_images = []
    gaps = []
    for i in range(len(boxes)):
        character_images.append(image[:, boxes[i][0]:boxes[i][1]])
        if i != 0:
            gaps.append(boxes[i][0] - boxes[i - 1][1])
    gap_indices = np.argsort(gaps)
    dash1 = min(gap_indices[-1], gap_indices[-2]) + 1
    dash2 = max(gap_indices[-1], gap_indices[-2]) + 2

    return character_images, dash1, dash2, True


def get_recognized_chars(images, plate, dot1, dot2):
    """Recognizes the characters in the plate at different vertical positions only if segmentation was done via
    segmentation_alternative and returns the characters and scores that had the best scores."""

    final_characters = []
    final_scores = [float('inf')] * 6

    # we assumed each character has a height of approximately 18% of the plates width
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
                final_characters = recognized_chars
                final_scores = character_scores
    # returns array of all the (hopefully 6) recognized chars
    return final_characters, final_scores


def recognize_characters(images, dot1, dot2, start_index, end_index):
    """Recognize separate character at given indices."""

    chars = []  # this list will contain the 6 characters found
    char_scores = []  # this list will contain the scores per each character found
    # split points to make sure the three, by dashes seperated, parts of the plate are treated independently
    split_points = [0, dot1, dot2 - 1, len(images)]
    # find characters seperate per each third
    for j in range(0, 3):
        numbers = []
        letters = []
        num_scores = []
        let_scores = []
        for jj in range(len(images)):
            # choose images belonging to one of the three parts determined by j
            if split_points[j] <= jj < split_points[j + 1]:
                # crop correct image at certain height and then to bounding box
                cropped = crop_to_bounding_box(images[jj][start_index:end_index])
                # make sure no errors occur
                if len(cropped) < 2 or len(cropped[0]) < 2:
                    break

                # get the character with the best score
                updateScores(cropped, let_scores, letters, num_scores, numbers)
        # for all characters found in the current third, choose either only numbers or only letters,
        # depending on the lowest score
        if sum(num_scores) < sum(let_scores):
            # Append to final result
            char_scores += num_scores
            chars += numbers
        else:
            # Append to final result
            char_scores += let_scores
            chars += letters
    return chars, char_scores


def updateScores(cropped, let_scores, letters, num_scores, numbers):
    """Performs updating of scores. Extracted as a refactoring to avoid duplication."""

    number, score1 = give_label_two_scores(cropped, True)
    letter, score2 = give_label_two_scores(cropped, False)
    numbers.append(number)
    num_scores.append(score1)
    letters.append(letter)
    let_scores.append(score2)


def give_label_two_scores(test_image, is_digit):
    """ Returns a label for a given test character image.

    Stores the separate difference scores between the test image and all reference characters.
    Next, it picks the lowest 2 scores and checks how similar the scores are.
    If the ratio of the scores is not below a certain threshold, the result is ambiguous.
    A choice is then made based off the euclidean distance between the sift descriptors of the picked
    reference characters and the sift descriptor of the test image.
    """

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
    if B == 0 or (1 - EPSILON < A / B < 1 + EPSILON):
        our_sift = sift_descriptor(test_image)
        sift_1 = sift_descriptor(reference_characters[result_char_1])
        sift_2 = sift_descriptor(reference_characters[result_char_2])
        diff_1 = np.linalg.norm(sift_1 - our_sift)
        diff_2 = np.linalg.norm(sift_2 - our_sift)
        if diff_1 > diff_2:
            return result_char_2, A

    return result_char_1, A


def difference_score(test_image, reference_character):
    """ Performs bitwise XOR to count the number of non-zero pixels. """

    reference_character = cv2.resize(reference_character, (len(test_image[0]), len(test_image)))
    # return the number of non-zero pixels
    return np.count_nonzero(cv2.bitwise_xor(test_image, reference_character))


def sift_descriptor(image):
    """ Sift descriptor to find descriptor vector of length 128. """

    image = cv2.resize(image, (16, 16))  # 30,20
    N = 4
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


def get_gradient(image):
    """ Finds the gradient magnitude and gradient orientation of the given image. """

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


def get_from_contours(images, dot1, dot2):
    """Recognizes the characters in the plate only if segmentation was done via contours."""

    chars = []  # this list will contain the 6 characters found
    char_scores = []  # this list will contain the scores per each character found
    # split points to make sure the three, by dashes seperated, parts of the plate are treated independently
    split_points = [0, dot1, dot2 - 1, len(images)]
    # find characters seperate per each third
    for j in range(0, 3):
        numbers = []
        letters = []
        num_scores = []
        let_scores = []
        for jj in range(len(images)):
            # choose images belonging to one of the three parts determined by j
            if split_points[j] <= jj < split_points[j + 1]:
                # crop correct image to bounding box
                cropped = crop_to_bounding_box(images[jj])
                # make sure no errors occur
                if len(cropped) < 2 or len(cropped[0]) < 2:
                    break

                # get the character with the best score
                updateScores(cropped, let_scores, letters, num_scores, numbers)
        # for all characters found in the current third, choose either only numbers or only letters, depending on the lowest score
        if sum(num_scores) < sum(let_scores):
            # append to final result
            char_scores += num_scores
            chars += numbers
        else:
            # append to final result
            char_scores += let_scores
            chars += letters
    return chars, char_scores


def crop_to_bounding_box(cropped):
    """Crop given image to boundary box."""

    [X, Y, W, H] = cv2.boundingRect(cropped)
    cropped = cropped[Y:Y + H, X:X + W]
    return cropped


def format_plate(plate):
    """Format plate so that it can be correctly parsed by evaluator."""

    final_plate = ''
    for char in plate:
        final_plate += char
    return final_plate
