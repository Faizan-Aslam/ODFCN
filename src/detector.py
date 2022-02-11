import cv2
import numpy as np
import math

def detector(img_path):
    captured_image = cv2.imread(img_path)

    lower_red = np.array([0,10,20])
    upper_red = np.array([15,255,255])
    mask0 = cv2.inRange(captured_image, lower_red, upper_red)
    final_mask = mask0
    ## slice the red
    imask = final_mask > 0
    red = np.zeros_like(captured_image, np.uint8)
    red[imask] = captured_image[imask]

    image = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # --- create a black image to see where those corners occur ---
    mask = np.zeros_like(gray)
    # --- applying a threshold and turning those pixels above the threshold to white ---
    mask[dst > 0.01 * dst.max()] = 255
    # cv.imshow('mask', mask)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]   # --- [0, 0, 255] --> Red ---
    # print(captured_image.shape)
    coordinates = np.argwhere(mask)
    # Convert array of arrays to lists of lists
    coordinates_list = [l.tolist() for l in list(coordinates)]
    # Convert list to tuples
    coordinates_tuples = [tuple(l) for l in coordinates_list]
    # Create a distance threshold
    thresh =10
    # Compute the distance from each corner to every other corner.
    def distance(pt1, pt2):
        (x1, y1), (x2, y2) = pt1, pt2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist    
    # Keep corners that satisfy the distance threshold
    coordinates_tuples_copy = coordinates_tuples
    i = 1
    for pt1 in coordinates_tuples:
        for pt2 in coordinates_tuples[i::1]:
            if (distance(pt1, pt2) < thresh):
                coordinates_tuples_copy.remove(pt2)
        i += 1
    l2 = [tuple[::-1] for tuple in coordinates_tuples]
    arr = np.array(l2)
    x_coordinates = np.average(arr, axis=0)[0]
    y_coordinates = np.average(arr, axis=0)[1]
    return x_coordinates, y_coordinates

