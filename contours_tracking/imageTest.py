import cv2
import imutils

raw_image = cv2.imread('contorus.png')
cv2.imshow('Original Image', raw_image)
# cv2.waitKey(0)

bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
cv2.imshow('Bilateral', bilateral_filtered_image)
# cv2.waitKey(0)

edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
cv2.imshow('Edge', edge_detected_image)
# cv2.waitKey(0)



# edged is the edge detected image
cnts = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
# loop over the contours
contour_list = []
for c in cnts:
    # approximate the contour
    print(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    area=cv2.contourArea(c)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if ((len(approx) > 9) & (len(approx) < 23) & (area > 30) ):
        contour_list.append(c)
cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
cv2.imshow('Objects Detected',raw_image)
cv2.waitKey(0)