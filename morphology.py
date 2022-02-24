import cv2
import matplotlib as plt
import numpy as np

AMBIGUOUS_RESULT = "AMBIGUOUS"
EPSILON = 0.15


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def loadImage(filepath, filename, grayscale=True):
    return cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def makeGrayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def opening(img, structuring_element):
    return cv2.dilate(cv2.erode(img, structuring_element), structuring_element)


def closing(img, structuring_element):
    return cv2.erode(cv2.dilate(img, structuring_element), structuring_element)


# !The test_image and reference_character must have the same shape
def difference_score(test_image, reference_character):
    # xor images
    # plotImage(test_image, "Test")
    # plotImage(reference_character, "Reference")
    bitwiseXOR = cv2.bitwise_xor(test_image, reference_character)
    result = reference_character - bitwiseXOR
    # plotImage(result,"XOR")

    # return the number of non-zero pixels
    return np.count_nonzero(bitwiseXOR)


def give_label_lowest_score(test_image):
    # Get the difference score with each of the reference characters
    # (or only keep track of the lowest score)
    lowest = float('inf')
    result = 0
    for char in reference_characters:
        temp = difference_score(test_image, reference_characters[char])
        if temp < lowest:
            lowest = temp
            result = char

    # Return a single character based on the lowest score
    return result


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

    #     total = 0
    #     for i in reference_characters[str(result_char)]:
    #         total = total + len(i)

    ratio = A / B
    if ratio > 1 - EPSILON and ratio < 1 + EPSILON:
        return AMBIGUOUS_RESULT
    # Return a single character based on the lowest score
    return result_char


def main():
    # TODO capture image from video
    cap = cv2.VideoCapture("TrainingSet/Categorie I/Video2_2.avi")

    # Choose a frame to work on
    frameN = 42

    for i in range(0, frameN):
        # Read the video frame by frame
        ret, frame = cap.read()
        # if we have no more frames end the loop
        if not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Load the reference characters
    character_set = {'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                     '3', '4', '5', '6', '7', '8', '9'}
    # letter_set = {'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z'}
    # number_set = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    reference_characters = {}
    letter_counter = 0
    number_counter = 0
    for char in character_set:
        if char.isdigit():
            reference_characters[char] = loadImage("SameSizeLetters/", str(number_counter) + ".bmp")
            number_counter = number_counter + 1
        else:
            reference_characters[char] = loadImage("SameSizeNumbers/", str(letter_counter) + ".bmp")
            letter_counter = letter_counter + 1

    # Load the test set - they are named test_N where N is a number
    # test_images = []
    # for i in range(1, 14):
    #     image_name = "Video" + ("0" if (i < 10) else "") + str(i) + ".png"
    #     test_images.append(loadImage("TrainingSet/Categorie I", image_name))

    # test_image = loadImage("TrainingSet/Categorie I", "Video2_2.avi")

    # Apply your algorithm
    # for img in test_images:
    #     plotImage(img, give_label_two_scores(img))

    # plotImage(test_image, give_label_two_scores(test_image))
    plotImage(frame, "Test")


main()
