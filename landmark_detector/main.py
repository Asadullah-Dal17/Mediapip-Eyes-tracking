import cv2 as cv 
import mediapipe as mp
import csv 
file_path = 'file.csv'
with open(file_path, 'r') as csv_file:
    data_list= list(csv.reader(csv_file))
    LEFT_EYE =[int(i) for i in data_list[1][1:]]
    RIGHT_EYE = [int(i) for i in data_list[0][1:]]
    print(RIGHT_EYE)
mp_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture("Video.mp4")
# land marks extarctor function 
def landmarks_detector(image, results, draw=False):
    """
    image : mat object(numpy array image) in RGB formate, instead of OpeCV formate which is BGR
    results: These are landmark return by Mediapipe module, 
    return: image, and All the landmarks list, have coordinates in tuple(x,y) and ID of each landmarks as integer value
    """
    # creating empty list 
    mesh_points_list =[]
    # getting the width and height
    height, width = image.shape[:2]
    for Ids ,marks in enumerate(results.multi_face_landmarks[0].landmark):
        # adding land mark to list with its id or indes number
        mesh_points_list.append([Ids, (int(marks.x*width), int(marks.y*height))])
        if draw==True:
            cv.circle(image,(int(marks.x*width), int(marks.y*height)), 2, (0,0,255),-1)
    # return the image and point of landmarks all 
    return image, mesh_points_list

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
        if results.multi_face_landmarks:
            image, points = landmarks_detector(rgb_frame, results)
            [cv.circle(frame, (points[pos][1]), 1,(0,0,255), -1) for pos in LEFT_EYE]
            [cv.circle(frame, (points[pos][1]), 1,(255,0,255), -1) for pos in RIGHT_EYE]
        cv.imshow('camera', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()