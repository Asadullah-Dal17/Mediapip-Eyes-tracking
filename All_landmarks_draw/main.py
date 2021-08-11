from typing import Counter
import cv2 as cv 
import mediapipe as mp
import csv 
import utils


LIPS=[61,146, 91, 181,181,84,17, 314, 405, 321, 375, 61, 185,40,39,37,0,267,269,270,409,78,95,88,178,87,14,317,402,318,324,78,191,80,81,82,13,312,311,310,415]
LEFT_EYE =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
LEFT_EYEBROW =[336,296,334,293,300,276,283,282,295,285]
RIGHT_EYE=[33, 7 , 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
Right_Eyes = [70,63,105,66,65, 52,53,46]  
RIGHT_EYEBROW=[70,63,105,66,107,55,65,52,53,46]

FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,58, 132,93, 234, 127, 162,21,54, 103,67, 109]
mp_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture("Girl.mp4")
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
    cont =0
    while True:
        ret, frame = camera.read()
        if ret is False:
            break
        # frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        # converting color space from BGR to RGB 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # getting the frame width 
        height, width = frame.shape[:2]
        # print(width, height)
        rgb_frame.flags.writeable=False
        results = face_mesh.process(rgb_frame)
        frame=utils.rectTrans(frame, pt1=(30, 320), pt2=(230, 320+200), color=(0,255,255),thickness=-1, opacity=0.4)
        utils.textBlurBackground(frame, 'Blured Background Text', cv.FONT_HERSHEY_COMPLEX, 0.8, (60, 140),2, utils.GREEN, (71,71), 13, 13)
        frame=utils.textWithBackground(frame, 'Colored Background Texts', cv.FONT_HERSHEY_SIMPLEX, 0.8, (60,80), textThickness=2, bgColor=utils.GREEN, textColor=utils.BLACK, bgOpacity=0.7, pad_x=6, pad_y=6)
        # imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if results.multi_face_landmarks:
            image, points = landmarks_detector(rgb_frame, results)
            # face = [points[pos] for pos in silhouette]
            # merge= leftEyeUpper0+leftEyeLower0
            # print(merge)
            lips_mark  = [points[pos] for pos in LIPS]
            right_eye_marks = [points[pos] for pos in RIGHT_EYE]
            right_eyebrow_marks = [points[pos] for pos in RIGHT_EYEBROW]
            # print(points)
            # for t in points:
                # print(t[0])
            # for i, p in enumerate(points):
            #     # print(p)
            #     cv.circle(frame, p[0],3, utils.ORANGE, -1 )
            #     cv.putText(frame, f'{i}', p[0],cv.FONT_HERSHEY_COMPLEX,0.7, utils.YELLOW, 2 )
            left_eye_marks = [points[pos] for pos in LEFT_EYE]
            left_eyebrow_marks = [points[pos] for pos in LEFT_EYEBROW]
            face_oval_marks = [points[pos] for pos in FACE_OVAL]

            frame = utils.fillPolyTrans(frame, face_oval_marks, utils.BLUE, 0.2)
            frame = utils.fillPolyTrans(frame, right_eye_marks, utils.YELLOW, 0.3)
            frame = utils.fillPolyTrans(frame, right_eyebrow_marks, utils.CYAN, 0.3)
            frame = utils.fillPolyTrans(frame, left_eye_marks, utils.YELLOW, 0.3)
            frame = utils.fillPolyTrans(frame, left_eyebrow_marks, utils.CYAN, 0.3)
            frame = utils.fillPolyTrans(frame, lips_mark, utils.ORANGE, 0.3)
            # frame = utils.fillPolyTrans(frame, face, utils.GREEN, 0.3)
            # frame = utils.fillPolyTrans(frame, right, utils.MAGENTA, 0.5)
        cv.imwrite(f'image/landmark{cont}.png',frame)
        cont+=1
        cv.imshow('camera', frame)
        # break
        key = cv.waitKey(10)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()