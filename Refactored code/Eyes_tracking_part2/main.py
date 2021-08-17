import cv2 as cv 
import numpy as np 
import mediapipe as mp
import utils

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
fonts =cv.FONT_HERSHEY_COMPLEX
# Frame per seconds
frame_counter =0
CLOSED_EYE_FRAME =3
TOTAL_BLINKS =0
COUNTER_WHILE_CLOSED =0

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

def blinkRatio(img, landmark,right_eye, left_eye):
    
    # width of eye,in pixels or horizontal line.
    
    # Right Eye 
    right_hx, right_hy = landmark[right_eye[0]]
    right_hx1, right_hy1 = landmark[right_eye[8]]
    right_eye_pixel_width = right_hx1-right_hx
    
    # Left Eyes
    left_hx, left_hy = landmark[left_eye[0]]
    left_hx1, left_hy1 = landmark[left_eye[8]]
    left_eye_pixel_width = left_hx1-left_hx

    # ---------------------------------------
    # vertical line or height of eyes
    
    # Right Eyes 
    right_vx, right_vy = landmark[right_eye[12]]
    right_vx1, right_vy1 = landmark[right_eye[4]]
    right_eye_pixel_height = right_vy1 - right_vy
    
    # Left Eye
    left_vx, left_vy = landmark[left_eye[12]]
    left_vx1, left_vy1 = landmark[left_eye[4]]
    left_eye_pixel_height = left_vy1 - left_vy
    # ---------------------------------------


    img =utils.textBlurBackground(img, f'w: {left_eye_pixel_width}  h: {left_eye_pixel_height}', fonts, 1.2, (50,50), 2,(0,255,255), (99,99))
    
    right_ratio = right_eye_pixel_width/right_eye_pixel_height
    
    left_ratio = left_eye_pixel_width/left_eye_pixel_height
    eyes_ratio = (left_ratio + right_ratio) / 2
    return eyes_ratio

# setting up camera 
cap = cv.VideoCapture(3)

# configring mediapipe for face mesh detection
with map_face_mesh.FaceMesh( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as face_mesh:
    # string video/webcame feed here     
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_AREA)
        
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
            right_eye_cord = [mesh_cords[point] for point in RIGHT_EYE]
            left_eye_cord = [mesh_cords[point] for point in LEFT_EYE]
            
            # drawing the eyes 
            # frame =utils.fillPolyTrans(frame, left_eye_cord, utils.PINK, 0.6)
            # frame =utils.fillPolyTrans(frame, right_eye_cord, utils.PINK, 0.6)

            eyes_ratio =blinkRatio(frame,mesh_cords, RIGHT_EYE, LEFT_EYE)

            if eyes_ratio>5:
                COUNTER_WHILE_CLOSED +=1
                frame = utils.textWithBackground(frame, "Blink", fonts, 1.7, (100,100), 2, utils.YELLOW, pad_x=9, pad_y=9, bgOpacity=0.8)
                
                print('BLINK')
            else:
                if COUNTER_WHILE_CLOSED >CLOSED_EYE_FRAME:
                    COUNTER_WHILE_CLOSED =0
                    TOTAL_BLINKS +=1
            print(TOTAL_BLINKS)
        cv.imshow('frame',frame)
        
        key = cv.waitKey(1)
        if key==ord('q'):
            break
cv.destroyAllWindows()
cap.release()
