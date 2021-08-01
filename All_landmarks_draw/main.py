import cv2 as cv 
import mediapipe as mp
import csv 
file_path = 'face_mesh.csv'
with open(file_path, 'r') as csv_file:
    data_list= list(csv.reader(csv_file))
    silhouette =[int(i) for i in data_list[0][1:]]
    lipsUpperOuter = [int(i) for i in data_list[1][1:]]

lipsLowerOuter = [int(i) for i in data_list[2][1:]]
lipsUpperInner = [int(i) for i in data_list[3][1:]]
lipsLowerInner = [int(i) for i in data_list[4][1:]]
rightEyeUpper0 = [int(i) for i in data_list[5][1:]]
rightEyeLower0 = [int(i) for i in data_list[6][1:]]
rightEyeUpper1 = [int(i) for i in data_list[7][1:]]
rightEyeLower1 = [int(i) for i in data_list[8][1:]]
rightEyeUpper2 = [int(i) for i in data_list[9][1:]]
rightEyeLower2 = [int(i) for i in data_list[10][1:]]
rightEyeLower3 = [int(i) for i in data_list[11][1:]]
rightEyebrowUpper = [int(i) for i in data_list[12][1:]]
rightEyebrowLower = [int(i) for i in data_list[13][1:]]
leftEyeUpper0 = [int(i) for i in data_list[14][1:]]
leftEyeLower0 = [int(i) for i in data_list[15][1:]]
leftEyeUpper1 = [int(i) for i in data_list[16][1:]]
leftEyeLower1 = [int(i) for i in data_list[17][1:]]
leftEyeUpper2 = [int(i) for i in data_list[18][1:]]
leftEyeLower2 = [int(i) for i in data_list[19][1:]]
leftEyeLower3 = [int(i) for i in data_list[20][1:]]
leftEyebrowUpper = [int(i) for i in data_list[21][1:]]
leftEyebrowLower = [int(i) for i in data_list[22][1:]]
midwayBetweenEyes = [int(i) for i in data_list[23][1:]]
'''noseTip = [int(i) for i in data_list[1][1:]]
noseBottom = [int(i) for i in data_list[1][1:]]
noseRightCorner = [int(i) for i in data_list[1][1:]]
noseLeftCorner = [int(i) for i in data_list[1][1:]]
rightCheek = [int(i) for i in data_list[1][1:]]
leftCheek = [int(i) for i in data_list[1][1:]]'''
mp_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture(0)
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
            [cv.circle(frame, (points[pos][1]), 2,(0,0,255), -1) for pos in rightEyebrowLower]
            [cv.circle(frame, (points[pos][1]), 2,(255,0,255), -1) for pos in rightEyebrowLower]
        cv.imshow('camera', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()