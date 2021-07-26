import cv2 as cv 
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = camera.read()
        if ret is False:
            break
        # converting color space from BGR to RGB 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # getting the frame width 
        width, height = frame[:2]
        rgb_frame.flags.writeable=False
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
             print(type(results.multi_face_landmarks))


            #  break


            # for face_marks in results.multi_face_landmarks:
            #     # print(face_marks)
            # break

        cv.imshow('camera', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()
