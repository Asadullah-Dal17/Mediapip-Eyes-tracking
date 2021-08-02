import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500,450))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        roi = frame[50:500, 300:800]
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (11, 11), 0)
        gray_roi = cv2.medianBlur(gray_roi, 3)
# cv2.contourArea
        threshold = cv2.threshold(gray_roi, 15, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours_list =[]
        for contour in contours:
       # approximate the contour
            print(contour)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)
            if (len(approx)>8) and len(approx)<23 and area>30:
                contours_list.append(contour)
        cv2.drawContours(roi, contours_list, -1, (255,0,0), 2)
            
        cv2.imshow("roi", roi)
        cv2.imshow('Threshold', threshold)
        # cv2.imshow('gray_roi', gray_roi)
        out.write(roi)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    if cv2.waitKey(15) & 0xFF == ord('q'): # Press 'Q' on the keyboard to exit the playback
        break
cap.release()
out.release()
cv2.destroyAllWindows()