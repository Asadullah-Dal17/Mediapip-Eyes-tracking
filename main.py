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
        height, width = frame.shape[:2]
        # print(width, height)
        rgb_frame.flags.writeable=False
        results = face_mesh.process(rgb_frame)
        mesh_points_list =[]
        if results.multi_face_landmarks:
            for Ids ,marks in enumerate(results.multi_face_landmarks[0].landmark):
                # adding land mark to list with its id or indes number
                mesh_points_list.append([Ids, (int(marks.x*width), int(marks.y*height))])
                cv.circle(frame,(int(marks.x*width), int(marks.y*height)), 2, (0,0,255),-1)
 
        
        cv.imshow('camera', frame)
        
        # print(mesh_points_list[0])
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()
