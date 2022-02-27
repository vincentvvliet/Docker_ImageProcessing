import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from Localization import find_plate
from Recognize import segment_and_recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""
INVALID = [601, 769, 913, 985, 1129, 1177, 1225, 1321, 1369, 1417, 1441, 1465, 1513, 1561, 1609]
results = []


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    global results
    done_with_car = False
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = -1
    start = datetime.now()

    # Read until video is completed
    while cap.isOpened():

        count += 1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Frame skipping s.t. frames are not on boundary of interval in evaluator
            if (count - 1) % 6 == 0 and count > 0:
                print(count)
                plate, found = find_plate(frame)
                if not found or len(plate) < 5 or len(plate[0]) < 100:
                    # TODO add comment
                    continue

                # Call pipeline
                plate, score = segment_and_recognize(plate, found)

                if plate != '':
                    # If plate is successfully non-empty, add to result
                    results.append((plate, score, count))

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # Write to file
    write(convert_to_single_plate(results), save_path)
    print("Script completed in:", datetime.now() - start)


def convert_to_single_plate(arr):
    """Convert multiple instances of a plate to a single plate.

    Recognition pipeline adds all found plates to a result data structure.
    This function finds which instances correspond to the same plate.
    Finally, it returns an array of the most common plates.
    """

    result = []
    doubt = []
    doubt_scores = []
    same_car = []
    same_car_scores = []
    first_frame = 0
    last_frame = 0
    i = 0
    while i < len(arr):
        plate = arr[i][0]
        score = arr[i][1]
        frame = arr[i][2]
        if len(same_car) == 0:
            first_frame = frame
            last_frame = frame
            same_car.append(plate)
            same_car_scores.append(score)
            i += 1
        else:
            if sum(1 for a, b in zip(same_car[-1], plate) if a != b) < 4:
                same_car += doubt + [plate]
                same_car_scores += doubt_scores + [score]
                doubt = []
                doubt_scores = []
                last_frame = frame
                i += 1
            elif len(doubt) < 2:
                doubt.append(plate)
                doubt_scores.append(score)
                i += 1
            else:
                result.append(save_format(choose_plate(same_car, same_car_scores), first_frame, last_frame))
                i = i - len(doubt)
                doubt = []
                doubt_scores = []
                same_car = []
                same_car_scores = []

    if len(same_car) != 0:
        first_frame = arr[len(arr) - len(doubt) - len(same_car)][2]
        last_frame = arr[len(arr) - len(doubt) - 1][2]
        result.append(save_format(choose_plate(same_car, same_car_scores), first_frame, last_frame))

    if len(doubt) != 0:
        first_frame = arr[len(arr) - len(doubt)][2]
        last_frame = arr[-1][2]
        result.append(save_format(choose_plate(doubt, doubt_scores), first_frame, last_frame))

    return result


def save_format(plate, first_frame, last_frame):
    """Rewrites plate into to correct format for future parsing by evaluator."""

    result = [plate]
    frame = int(0.5 * (first_frame + last_frame))
    time = int(frame / 12)
    result.append(frame)
    result.append(time)
    return result


def choose_plate(same_car_plates, same_car_scores):
    """Choose the most common plate in an array of plates of the same car."""

    unique, counts = np.unique(same_car_plates, return_counts=True)
    best = max(counts)
    plates = []
    for i in range(len(counts)):
        if counts[i] == best:
            plates.append(unique[i])
    if len(plates) == 1:
        return plates[0]
    scores_per_plate = np.zeros(len(plates))
    for i in range(len(plates)):
        for j in range(len(same_car_plates)):
            if same_car_plates[j] == plates[i]:
                scores_per_plate[i] += np.sum(same_car_scores[j])

    return plates[np.argmin(scores_per_plate)]


def write(plates, save_path):
    """Write recognized plates to csv file."""

    # open the file in the write mode
    with open(save_path + '/Output.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # Add header
        header = ['License plate', 'Frame no.', 'Timestamp(seconds)']
        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)
