import cv2
import os
import matplotlib.pyplot as plt

THRESHOLD = 0.2  # Amount of pixels our result can be off by
label_path = "/home/imageprocessingcourse/Labeling/labels/"
result_path = "/home/imageprocessingcourse/Results/"
# label_path = "C:/Users/Vincent van Vliet/Desktop/TU/Y2/Q2/IP/Docker_ImageProcessing-updated/Docker_ImageProcessing/Labeling/labels"
# result_path = "C:/Users/Vincent van Vliet/Desktop/TU/Y2/Q2/IP/Docker_ImageProcessing-updated/Docker_ImageProcessing/Results"


def loadImage(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()

def main():
    # Define arrays with size of result set
    labelled = []
    result_image = []

    # Load all images
    label_images = os.listdir(label_path)
    for item in label_images:
        labelled.append(loadImage(label_path + item))

    result_images = os.listdir(result_path)
    for item in result_images:
        result_image.append(loadImage(result_path + item))

    comparison_result = []

    # Compare images
    for i, _ in enumerate(result_image):
        # Add parts that do not correspond to each other
        corresponding = cv2.bitwise_and(labelled[i], result_image[i])
        a = cv2.countNonZero(corresponding)
        b = cv2.countNonZero(labelled[i])
        ratio = a / b

        correct_result = False

        if ratio > 1 - THRESHOLD:
            correct_result = True

        comparison_result.append(correct_result)

    total = 65  # until frame_1536 is category I or II
    incorrect = 0
    for i in comparison_result:
        if i == False:
            incorrect += 1
    print("Error: ", incorrect)
    print("Correct: ", total - incorrect)
    print("Total: ", total)
    print("Score", (total - incorrect) / total)

main()

