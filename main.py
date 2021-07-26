import cv2 as cv 
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture(0)

while  True:
    ret, frame = camera.read()
    cv.imshow('camera', frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
camera.release()
