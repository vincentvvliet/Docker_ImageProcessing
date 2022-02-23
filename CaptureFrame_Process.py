import csv

import cv2
import matplotlib.pyplot as plt

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
same_car = []


def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    global same_car
    done_with_car = False
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    count = -1
    # Read until video is completed
    while cap.isOpened():

        count += 1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Frame skipping s.t. Category IV is skipped and frames are not on boundary of interval in evaluator
            if (count - 1) % 8 == 0 and count > 0:
                print(count)

                plate, found = find_plate(frame)
                if not found:
                    continue

                _, desc = sift.detectAndCompute(plate, None)
                compare = False

                if desc is not None:
                    if len(same_car) != 0:
                        # 2nd or more occurrence of plate
                        if (len(same_car[-1][0]) < 2 or len(desc) < 2):
                            # If not enough features for k=2 kNN match, then skip frame
                            continue
                        matches = flann.knnMatch(same_car[-1][0], desc, k=2)
                        ratio = 0.3
                        match_score = 0
                        for m, n in matches:
                            if m.distance < ratio * n.distance:
                                match_score += 1

                        # print("score:", match_score)
                        if match_score < 30 / 2.0:
                            # Different plate
                            # Only compare as soon as all frames of same plate have passed
                            compare = True
                            done_with_car = False
                            # Re-initialize for this plate
                            same_car = [[desc, match_score]]
                        else:
                            # Same plate, therefore append
                            same_car.append([desc, 30])
                            compare = False
                    else:
                        # Empty, therefore first occurrence of plate
                        same_car.append([desc, 30])

                # print("same_car size:", len(same_car))
                # if not done_with_car:
                done_with_car = segment_and_recognize(plate, found, count, compare)
                write(recognized_plates, save_path)

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
    # write(recognized_plates, save_path)


def write(plates, save_path):
    # open the file in the write mode
    with open(save_path + '/Output.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        header = ['License plate', 'Frame no.', 'Timestamp(seconds)']

        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)
