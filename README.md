# License Plate Recognition
License Plate Recognition project for CSE2225 Image Processing. This project makes use of OpenCV to read a video file containing different cars and runs an image processing pipeline to localize and recognize the letters from the license plates. The localization and recognition steps of the pipeline are explained below. The results are then converted to a csv file containing for each license plate the characters of the plate, the frame number at which it was recognized and the corresponding timestamp. The localization and recognition steps of the pipeline are seperately evaluated with localization_evaluator.py and recognition_evaluator.py respectively.

More details on the later changes made to the implementation can be seen in [our report](IP_Project_resit_report.pdf).

## Plate requirements
- Must be at least 200 pixels
- Must be more than 0.5% of the total image size
- Must be less than 20% of the total image size
- Must have a width to height ratio of less than 2

## Localization
1) Apply a colour mask and apply a bitwise and to segment these from the rest
2) Convert to grayscale and apply gaussian blur
3) Find all contours left in the image
4) Remove all irrelevant results in accordance with assignment
5) Choose the image with the ratio closest to the actual ratio of a license plate
6) Rotate the image to make plate level
7) Return resulting image

## Recognition
1) Check that localization was successful
2) Apply ISODATA thresholding to remove certain pixel values
3) Apply morphological operations to remove noise
4) Crop the image to the character height
5) Segment the image by characters
6) Find the best character based on the difference score between the character and the stored character
7) If this difference score gives an ambiguous result, find the best match using sift descriptors
8) Format the result in readable way

## Result
To get a final result, we compare multiple instances of the same plate into a single plate using the most common characters found. This implementation gave us a result of 85% accuracy based on the test data provided by the course. 

# Running project
To run the project, start Docker using the runDockerWindows.bat script.
Next, run `python main.py`

To evaluate the localization, run `localization_evaluator.py`

To evaluate the recognition, run `recognition_evaluator.py`

# Contributors
- Bram Stellinga
- Vincent van Vliet
