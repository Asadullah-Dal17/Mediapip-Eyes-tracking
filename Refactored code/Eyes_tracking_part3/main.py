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
frame_counter = 0 # for fps
CET_FRAMES = 3 # closed eyes threshold frames
TOTAL_BLINKS = 0 # total blinks counter 
CLOSED_EYES_FRAMAE_COUNTER = 0 # count frames while the eyes are closed.

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

#extract eyes from frame
def eyes_extractor(frame,right_eye_cords, left_eye_cords):
    
    # converting color image to Gray Scale image.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # getting the dimension of image
    dim = gray.shape 
    
    # creating empty mask
    mask = np.zeros(dim, dtype=np.uint8)

    left_np = np.array(left_eye_cord, dtype=np.int32)
    right_np = np.array(right_eye_cord, dtype=np.int32)
    # draw eyes portion on mask 
    cv.fillPoly(mask, [right_np], 255)

    cv.fillPoly(mask, [left_np], 255)
    cv.imshow('mask', mask)

    # spearting Eyes from frame 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[ mask==0 ] = 155

    # getting the min x,y and max x,y  of eyes 
    # Right Eye
    r_maxX = (max(right_eye_cord, key=lambda item: item[0]))[0]
    r_minX = (min(right_eye_cord, key=lambda item: item[0]))[0]
    r_maxY = (max(right_eye_cord, key=lambda item: item[1]))[1]
    r_minY = (min(right_eye_cord, key=lambda item: item[1]))[1]
    # Left Eye
    l_maxX = (max(left_eye_cord, key=lambda item: item[0]))[0]
    l_minX = (min(left_eye_cord, key=lambda item: item[0]))[0]
    l_maxY = (max(left_eye_cord, key=lambda item: item[1]))[1]
    l_minY = (min(left_eye_cord, key=lambda item: item[1]))[1]

    # cropping eyes from mask 
    # cropping right eye 
    cropped_right = eyes[r_minY:r_maxY, r_minX:r_maxX]
    # croping left eye 
    cropped_left = eyes[l_minY:l_maxY, l_minX:l_maxX]

    cv.imshow("left", cropped_left)
    cv.imshow('right', cropped_right)
    return cropped_right, cropped_left

def positionEstimator(frame, eye_image):
    # getting the width and height of eye image
    height, width = eye_image.shape[:2]
    
    # calculating the width of each piece 
    piece = int(width/3)
 
    # applying blur to remove some noise 
    gaussain_blur=cv.GaussianBlur(eye_image,(9,9), 0)
    blur = cv.medianBlur(gaussain_blur, 3)


    # applying thresholding to binarizing_image(black and white pixels only)
    ret, threshold_eye =cv.threshold(blur, 130,255, cv.THRESH_BINARY)
    
    # sliceing eye width into three pieces
    right_piece =threshold_eye[0:height, 0:piece]
    center_piece =threshold_eye[0:height, piece:piece+piece]
    left_piece =threshold_eye[0:height, piece+piece:width]
    
    eye_position, color =pixel_counter(right_piece, center_piece, left_piece)
    
    # cv.imshow('right', right_piece)
    # cv.imshow('center', center_piece)
    # cv.imshow('left', left_piece)
    # cv.imshow('gussain', threshold_eye)
    return eye_position, color


def pixel_counter(first_part, second_part, third_part):
    right_part = np.sum(first_part==0)
    center_part = np.sum(second_part==0)
    left_part = np.sum(third_part==0)
    eye_parts = [right_part, center_part, left_part]
    
    maxIndex =eye_parts.index(max(eye_parts))
    posEye = ''

    if maxIndex == 0:
        posEye = "RIGHT"
        colors = [utils.BLACK, utils.GREEN]
    elif maxIndex == 1:
        posEye = "CENTER"
        colors = [utils.YELLOW, utils.BLACK]
    elif maxIndex == 2:
        posEye = "LEFT"
        colors = [utils.PINK, utils.BLUE]
    else:
        posEye = "Eye Closed"
        colors = [utils.GRAY, utils.PURPLE]
    return posEye, colors

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
            right_eye, left_eye=eyes_extractor(frame, right_eye_cord, left_eye_cord)
            right_position, right_color =positionEstimator(frame,right_eye)
            frame = utils.textWithBackground(frame, f"{right_position}", fonts, 1.2, (70,200), 2,textColor=right_color[0], bgColor=right_color[1], pad_x=9, pad_y=9, bgOpacity=0.7)
            left_position, left_color =positionEstimator(frame,left_eye)
            frame = utils.textWithBackground(frame, f"{left_position}", fonts, 1.2, (70,300), 2,textColor=left_color[0], bgColor=left_color[1], pad_x=9, pad_y=9, bgOpacity=0.7)

            eyes_ratio =blinkRatio(frame,mesh_cords, RIGHT_EYE, LEFT_EYE)

            if eyes_ratio>5:
                CLOSED_EYES_FRAMAE_COUNTER +=1
                frame = utils.textWithBackground(frame, "Blink", fonts, 1.7, (100,100), 2, utils.YELLOW, pad_x=9, pad_y=9, bgOpacity=0.8)
            else:
                if CLOSED_EYES_FRAMAE_COUNTER >CET_FRAMES:
                    CLOSED_EYES_FRAMAE_COUNTER =0
                    TOTAL_BLINKS +=1
            # print(TOTAL_BLINKS)
            frame = utils.textWithBackground(frame, f"Total Blinks: {TOTAL_BLINKS}", fonts, 0.7, (20,150), 2, utils.GREEN, pad_x=4, pad_y=4, bgOpacity=0.8)

        cv.imshow('frame',frame)
        
        key = cv.waitKey(1)
        if key==ord('q'):
            break
cv.destroyAllWindows()
cap.release()
