import cv2 as cv 

video = cv.VideoCapture('video.mp4')



while True:
    ret, frame = video.read()
    if not ret:
        break
    
    roi = frame[50:500, 300:800]
    
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray_roi = cv.GaussianBlur(gray_roi, (11, 11), 0)
    gray_roi = cv.medianBlur(gray_roi, 3)
    edge =cv.Canny(gray_roi, 100, 255)
    threshold = cv.threshold(gray_roi, 15, 255, cv.THRESH_BINARY_INV)[1]
    cv.imshow("frame", roi)
    cv.imshow("edge", edge)
    cv.imshow('threshold', threshold)

    key = cv.waitKey(0)
    if key ==ord('q'):
        break
cv.destroyAllWindows()