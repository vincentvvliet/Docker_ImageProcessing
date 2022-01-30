import cv2
import os
import pandas as pd
import csv
from Localization import find_plate
from Recognize import segment_and_recognize
from Recognize import recognized_plates

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


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = -1
    # Read until video is completed
    while cap.isOpened():
        count += 1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Frame skipping s.t. Category IV is skipped and frames are not on boundary of interval in evaluator
            if (count - 1) % 24 == 0:
                print(count)

                # indices, angles, boxes = get_all_contours_info(frame)
                plate, found = find_plate(frame)
                segment_and_recognize(plate, found, count)

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
    write(recognized_plates, save_path)

def write(plates, save_path):
    # open the file in the write mode
    with open(save_path + '/Output.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        header = ['License plate', 'Frame no.', 'Timestamp(seconds)']

        writer.writerow(header)

        # write a row to the csv file
        writer.writerows(plates)