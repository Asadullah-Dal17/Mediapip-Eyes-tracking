import cv2 as cv 
import numpy as np 
import mediapipe as mp
import utils
import time
# Landmarks indices for different parts of face, mediapipe.

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 181, 84, 17, 314, 405, 321, 375, 61, 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 78, 95 ,88 ,178 ,87, 14, 317, 402, 318, 324, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415 ]

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# Variables 
# Frame per seconds
frame_counter =0

# Setting up mediapipe 
map_face_mesh= mp.solutions.face_mesh

# Different function
def faceLandmarksDetector(img, result, draw=False):

    # image width and height 
    img_height, img_width = img.shape[:2]

    # getting all the landmark  normalized cordinate(x,y) in the image
    # multiplying these coordinate with width and height we get image coordinates
    mesh_cord_point = [ (int(p.x*img_width) , int(p.y*img_height)) for p in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, point_cord, 2, utils.GREEN, -1) for point_cord in mesh_cord_point]
    # print()
    return mesh_cord_point

# setting up camera 
cap = cv.VideoCapture(0)

# configring mediapipe for face mesh detection
with map_face_mesh.FaceMesh( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as face_mesh:
    # string video/webcame feed here
    # initial time set here 
    starting_time =time.time()
    while True:
        ret, frame = cap.read()
        frame_counter +=1
        
        # converting color space from BGR to RGB 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # getting the frame width 
        height, width = frame.shape[:2]

        # getting the landmarks 
        results = face_mesh.process(rgb_frame)
        
        # checking if landmarks are detected or not 
        if results.multi_face_landmarks:
            # calling faceLandmarksDetector function and getting coordinate of each point in face mesh 
            mesh_cords =faceLandmarksDetector(img=frame, result=results)

            frame = utils.fillPolyTrans(img=frame,points=[mesh_cords[p] for p in FACE_OVAL],color=utils.BLACK, opacity=0.4)
            # draw lips landmarks portion 
            frame = utils.fillPolyTrans(img=frame,points=[mesh_cords[p] for p in LIPS],color=utils.WHITE, opacity=0.4)
            frame = utils.fillPolyTrans(img=frame,points=[mesh_cords[p] for p in LEFT_EYE],color=utils.YELLOW, opacity=0.4)
            frame = utils.fillPolyTrans(img=frame,points=[mesh_cords[p] for p in LEFT_EYEBROW],color=utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(img=frame,points=[mesh_cords[p] for p in RIGHT_EYE],color=utils.YELLOW, opacity=0.4)
            frame = utils.fillPolyTrans(img=frame,points=[mesh_cords[p] for p in RIGHT_EYEBROW],color=utils.GREEN, opacity=0.4)

        # calculating the end time of total frames 
        end_time = time.time()- starting_time

        # calculating the frames per seconds 
        fps = frame_counter/end_time
        
        
        cv.imshow('frame',frame)
        
        key = cv.waitKey(1)
        if key==ord('q'):
            break
cv.destroyAllWindows()
cap.release()
