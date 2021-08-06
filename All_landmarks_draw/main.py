import cv2 as cv 
import mediapipe as mp
import csv 
import utils


LIPS=[61,146, 91, 181,181,84,17, 314, 405, 321, 375, 61, 185,40,39,37,0,267,269,270,409,78,95,88,178,87,14,317,402,318,324,78,191,80,81,82,13,312,311,310,415]
LEFT_EYE =[263,249,390,373,374,380,381,382,263,466,388,387,386,385,384,398]
LEFT_EYEBROW =[276,283,282,295,300,293,334,296]
RIGHT_EYE=[33,7, 163,144,145,153,154,155,33,246,161,160159, 158, 157, 173]
RIGHT_EYEBROW=[46,53,52,65,70,63,105,66]
FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,58, 132,93, 234, 127, 162,21,54, 103,67, 109]
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
        mesh_points_list.append([(int(marks.x*width), int(marks.y*height))])
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
            face = [points[pos] for pos in silhouette]
            merge= leftEyeUpper0+leftEyeLower0
            print(merge)
            rightEyebrowL = [points[pos] for pos in [263,249,390,373,374,380,381,382,263,466,388, 387, 386, 385, 384, 398, 362]]
            
            


            
            # print(right)
            # print(silhouette)
            # [cv.circle(frame, (points[pos][1]), 2,(255,0,255), -1) for pos in rightEyebrowLower]
            # for point in points:
            #     cv.circle(frame, point, 2, (0, 255,0), -1)
            frame = utils.fillPolyTrans(frame, rightEyebrowL, utils.YELLOW, 0.5)
            # frame = utils.fillPolyTrans(frame, face, utils.GREEN, 0.3)


            # frame = utils.fillPolyTrans(frame, right, utils.MAGENTA, 0.5)
        cv.imshow('camera', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()