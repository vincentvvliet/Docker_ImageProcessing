import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Localization import find_plate
from Recognize import recognized_plates
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

    # sift = cv2.xfeatures2d.SIFT_create()
    # index_params = dict(algorithm=0, trees=5)
    # search_params = dict()
    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    count = -1
    # Read until video is completed
    while cap.isOpened():

        count += 1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Frame skipping s.t. Category IV is skipped and frames are not on boundary of interval in evaluator
            if (count - 1) % 9 == 0 and count > 0 and count < 1600:
                print(count)
                plate, found = find_plate(frame)
                if not found or len(plate) < 5 or len(plate[0]) < 100:
                    continue
                # _, desc = sift.detectAndCompute(plate, None)
                # compare = False

                # if desc is not None:
                #     if len(same_car) != 0:
                #         # 2nd or more occurrence of plate
                #         if (len(same_car[-1][0]) < 2 or len(desc) < 2):
                #             # If not enough features for k=2 kNN match, then skip frame
                #             continue
                #         matches = flann.knnMatch(same_car[-1][0], desc, k=2)
                #         ratio = 0.3
                #         match_score = 0
                #         for m, n in matches:
                #             if m.distance < ratio * n.distance:
                #                 match_score += 1

                #         # print("score:", match_score)
                #         if match_score < 30 / 2.0:
                #             # Different plate
                #             # Only compare as soon as all frames of same plate have passed
                #             compare = True
                #             done_with_car = False
                #             # Re-initialize for this plate
                #             same_car = [[desc, match_score]]
                #         else:
                #             # Same plate, therefore append
                #             same_car.append([desc, 30])
                #             compare = False
                #     else:
                #         # Empty, therefore first occurrence of plate
                #         same_car.append([desc, 30])

                # print("same_car size:", len(same_car))
                # if not done_with_car:
                plate, score = segment_and_recognize(plate, found, count, True)
                if plate != '':
                    results.append((plate,score,count))
                                

                # write(recognized_plates, save_path)

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
    result = into_single_plates(results)
    write(result,save_path)
    # write(recognized_plates, save_path)
def into_single_plates(arr):
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
                result.append(save_format(choose_plate(same_car,same_car_scores),first_frame,last_frame))
                i = i - len(doubt)
                doubt = []
                doubt_scores = []
                same_car = []
                same_car_scores = []


    if len(same_car) != 0:
        first_frame = arr[len(arr)-len(doubt)-len(same_car)][2]
        last_frame = arr[len(arr)-len(doubt)-1][2]
        result.append(save_format(choose_plate(same_car,same_car_scores),first_frame,last_frame))
    if len(doubt) != 0:
        first_frame = arr[len(arr)-len(doubt)][2]
        last_frame = arr[-1][2]
        result.append(save_format(choose_plate(doubt,doubt_scores),first_frame,last_frame))
            
    return result

def save_format(plate,first_frame,last_frame):
    result = [plate]
    frame = int(0.5*(first_frame+last_frame))
    time = int(frame/12)
    result.append(frame)
    result.append(time)
    return result


def choose_plate(same_car_plates, same_car_scores):
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
    # open the file in the write mode
    with open(save_path + '/Output.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        header = ['License plate', 'Frame no.', 'Timestamp(seconds)']

        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)
