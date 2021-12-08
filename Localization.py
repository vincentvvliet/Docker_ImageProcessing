import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""
def get_boundary_boxes(image):

    def rowhaswhite(row):
        for pixel in row:
            if pixel != 0:
                return True
        return False

    # split image horizontal and store both the index of the first row of each box and the index of the last row of each box
    boxes = []
    inwhite = False
    for i in range(0, len(image)):
        row = image[i]
        if inwhite:
            if not rowhaswhite(row):
                boxes[len(boxes)-1].append(i)
                inwhite = False
        else:
            if rowhaswhite(row):
                newbox = []
                newbox.append(i)
                boxes.append(newbox)
                inwhite = True
    
    # for each box, apply the same technique as above, but now in vertical direction and find the first and last column index
    for box in boxes:
        if len(box) < 2:
            box.append(len(image))
        inwhite = False
        for j in range(0, len(image[0])):
            allblack = True
            for i in range(box[0], box[1]):
                if image[i][j] != 0:
                    allblack = False
                    break
            if inwhite:
                if allblack:
                    box.append(j)
                    inwhite = False
            else:
                if not allblack:
                    box.append(j)
                    inwhite = True
        if len(box) == 2:
            box.append(0)
            box.append(0)
        if len(box)%2 != 0:
            box.append(len(image[0]))


    return boxes



def plate_detection(image):
    plate_imgs = image
    # Define color range
    colorMin = np.array([10, 60, 60])
    colorMax = np.array([26, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    hsi = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi, colorMin, colorMax)

    # Plot the masked image (where only the selected color is visible)
    # plotImage(mask, "Masked image", "gray")
    result = cv2.bitwise_and(plate_imgs, plate_imgs, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # delete some noise 
    result = cv2.erode(result, np.ones((10,10)))
    result = cv2.dilate(result, np.ones((10,10)))

    # get all the boundary boxes 
    boxes = get_boundary_boxes(result)

    # get edges of the boundary box which shape is the closest to the 52cm by 11cm dutch license plate
    widthdividedbyheight = float(52/11)
    edges = []
    difference = float('inf')
    for box in boxes:
        height = box[1]-box[0]
        index = 3
        while index < len(box):
            width = box[index] - box[index-1]
            diff = float(np.abs(float(widthdividedbyheight-float(width/height))))
            if diff < difference:
                difference = diff
                edges = [box[0], box[1], box[index-1], box[index]]
            index = index + 2


    # color the chosen boundary box green
    result = image
    for i in range(edges[0], edges[1]):
        result[i][edges[2]] = [0,255,0]
        result[i][edges[3]] = [0,255,0]
    for j in range(edges[2], edges[3]):
        result[edges[0]][j] = [0,255,0]
        result[edges[1]][j] = [0,255,0]
    



    # # Get coordinates of the plate
    # indices = []
    # for i, _ in enumerate(mask):
    #     for j, _ in enumerate(mask[i]):
    #         if mask[i][j] != 0:
    #             indices.append([i, j])
    # indices = sorted(indices, key=lambda indices: indices[0])
    # minXY = indices[0]
    # maxXY = indices[-1]
    # cropped = result[minXY[0]:maxXY[0], minXY[1]:maxXY[1]]

    # print(cropped[0][0])
    # # for row in cropped:
    # #     for pixel in row:
    # plotImage(cropped, "Cropped")

    return result


def plotImage(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()
