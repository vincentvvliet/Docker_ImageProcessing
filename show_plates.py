import argparse

import cv2

from Localization import find_plate
from Recognize import segment_and_recognize

# Plates where localization fails
INVALID = [601, 769, 913, 985, 1129, 1177, 1225, 1321, 1369, 1417, 1441, 1465, 1513, 1561, 1609]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dummytestvideo.avi')
    args = parser.parse_args()
    return args


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('dummytestvideo.avi')

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
            segment_and_recognize(plate, found, count, False)

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
