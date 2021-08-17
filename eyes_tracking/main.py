import cv2 as cv 
import mediapipe as mp
import csv 
import numpy as np 
import utils
# for the sound, to indicate the position of eyes 
from pygame import mixer
import pygame

mixer.init()
sound_left =mixer.Sound('left.wav')
sound_right = mixer.Sound('Right.wav')
sound_center =mixer.Sound ('center.wav')


font = cv.FONT_HERSHEY_COMPLEX
file_path = 'selected_landmarks.csv'
with open(file_path, 'r') as csv_file:
    data_list= list(csv.reader(csv_file))
    LEFT_EYE =[int(i) for i in data_list[1][1:]]
    RIGHT_EYE = [int(i) for i in data_list[0][1:]]
    # print(RIGHT_EYE)
mp_face_mesh = mp.solutions.face_mesh


# land marks extarctor function 
def landmarks_detector(image, results, draw=False):
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

def blink_ratio(frame,landmarks, eye_points, right_eye=False):
    # horizontal line data 
    hx, hy = landmarks[eye_points[0]][1]
    hx1, hy1 = landmarks[eye_points[7]][1]
    # Top_Vertical Points 
    second_x,vy11 = landmarks[eye_points[12]][1]
    first_x,vy = landmarks[eye_points[11]][1]
    vx1, vy1 = landmarks[eye_points[4]][1]
    
    if right_eye:
        second_x,vy11 = landmarks[eye_points[13]][1]
        first_x,vy = landmarks[eye_points[12]][1]
        # cv.circle(frame, ( landmarks[eye_points[13]][1] ), 2,(0,255,0), -1)
        # cv.circle(frame, ( landmarks[eye_points[12]][1] ), 2,(0,0,255), -1) 
        padding = int((first_x- second_x)/2)
        # cv.line(frame, (vx1, vy1), (second_x+padding,vy ), (0,0,255),2)
    else:
        padding = int((first_x -second_x)/2)
        # cv.line(frame, (vx1, vy1), (second_x+padding,vy ), (0,255,255))
   
    # cv.line(frame, (hx, hy), (hx1, hy1), (0,255,0),2)
    eye_pixel_with = hx1-hx 
    # print(h_px)
    eye_pixel_height = vy1 -vy
    ratio = eye_pixel_with/ eye_pixel_height

    return ratio
def extracting_eyes(frame, landmarks, left_eye, right_eye):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    right_points = [landmarks[i][1] for i in left_eye]
    left_points = [landmarks[i][1] for i in right_eye]

    # height, width = frame.shape[:2]
    mask = np.zeros((frame.shape), dtype=np.uint8)
    #left eye min and max for croping
    maxX = (max(left_points, key=lambda item: item[0]))[0]
    minX = (min(left_points, key=lambda item: item[0]))[0]
    maxY = (max(left_points, key=lambda item: item[1]))[1]
    minY = (min(left_points, key=lambda item: item[1]))[1]
    #Right min and max for croping the rectangle 
    r_max_x = (max(left_points, key=lambda item: item[0]))[0]
    r_min_x = (min(left_points, key=lambda item: item[0]))[0]
    r_max_y = (max(left_points, key=lambda item: item[1]))[1]
    r_min_y = (min(left_points, key=lambda item: item[1]))[1]
    # print(maxX)
    
    left_np = np.array(left_points, dtype=np.int32)
    right_np = np.array(right_points, dtype=np.int32)
    # print(left_points)

    # Fill the Region of Eye on the mask
    cv.fillPoly(mask, [right_np], 255)
    cv.fillPoly(mask, [left_np], 255)
    eye = cv.bitwise_and(frame, frame, mask=mask)
    eye[mask==0]=255
    left_eye_cropped =eye[minY:maxY, minX:maxX]
    
    right_eye_cropped = eye[r_min_y:r_max_y, r_min_x:r_max_x]
    # cv.imshow('eye', left_eye_cropped)
    cv.imshow('right eye', right_eye_cropped)
    cv.imshow('mask', eye)
    return right_eye_cropped, left_eye_cropped
    
def position_estimator(frame,eye_image):
    eye_height, eye_width = eye_image.shape[:2]
    portion = int(eye_width/3)
    # print("portion: " ,portion, "total with: ", eye_width)
    # blur = cv.GaussianBlur(eye_image, 100, 200)
    blur = cv.GaussianBlur(eye_image, (9, 9), 0)
    blur = cv.medianBlur(blur, 3)
    ret, threshed_eye = cv.threshold(blur, 130, 255, cv.THRESH_BINARY)
    first_part= threshed_eye[0:eye_height, 0:portion]
    second_part = threshed_eye[0:eye_height, portion:portion+portion]
    third_part = threshed_eye[0:eye_height, portion+portion:eye_width]
    images = np.hstack((second_part, first_part,third_part))
    cv.imshow('stack', images)
    cv.imshow('threshed', threshed_eye)
    estimated_pos , colors =pixel_counter(first_part, second_part, third_part)
    return estimated_pos, colors
# Pixel Counter Function
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

    
camera = cv.VideoCapture(0) 
counter=0
last_count=0
last_ratio =0
right_cond = False
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as face_mesh:
    while True:
        ret, frame = camera.read()
        
        if ret is False:
            break
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        # converting color space from BGR to RGB 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # getting the frame width 
        height, width = frame.shape[:2]
        # print(width, height)
        rgb_frame.flags.writeable=False
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            image, points = landmarks_detector(rgb_frame, results)
            # Draw the all Landmarks 
            # [cv.circle(frame, land[1],2, (0,250,50),-1) for land in points]
            #Drwing All Eyes Points 
            # [cv.circle(frame, (points[pos][1]), 1,(0,0,255), -1) for pos in LEFT_EYE]
            # [cv.circle(frame, (points[pos][1]), 1,(255,0,255), -1) for pos in RIGHT_EYE]
            
            # Eyes Tracking 
            right_cropped, left_cropped =extracting_eyes(frame, points, LEFT_EYE, RIGHT_EYE)
            # eye_position_estimator(frame, points, LEFT_EYE)
            pos_estimation, color = position_estimator(frame, left_cropped)
            # print(pos_estimation)
            frame =utils.textWithBackground(frame,f'Pos: {pos_estimation}',font, 0.7,(20, 30),2, color[1], color[0],6, 6, 0.7)
            # cv.putText(frame, f"Pos: {pos_estimation} ", (100, 100),font, 0.7, (0,0,0),2 )
            # position_estimator(left_cropped)
            print(pygame.mixer.get_busy())
            
            if pos_estimation=="RIGHT" and pygame.mixer.get_busy()==0: sound_right.play()
            if pos_estimation=="CENTER" and pygame.mixer.get_busy()==0: sound_center.play()
            if pos_estimation=="LEFT" and pygame.mixer.get_busy()==0: sound_left.play()



            # Eyes Blinking Detector 
            right_ratio=blink_ratio(frame ,points, RIGHT_EYE, right_eye=True)
            left_ratio=blink_ratio(frame ,points, LEFT_EYE)
            ratio = (right_ratio +left_ratio)/2
            # cv.putText(frame, f'Ratio: {round(ratio,3)}', (40,40), font, 0.5, (0,255,0),2 )
            # utils.textBlurBackground(frame, f'Ratio:  {round(ratio,3)}', font, 0.7, (30, 30), 2, (0,255,0), (71,71), 5,5)
            cv.putText(frame, f'Last_ratio: {round(last_ratio,3)} :: last count: {last_count}', (40,55), font, 0.5, (0,255,0),2 )
            if ratio>5:
                counter +=1
                last_count= counter
                last_ratio = ratio
                print("blink")
                cv.putText(frame, "BLINK :) ", (70, 70), font, 0.7, (0, 255,0), 2)
            else: 
                counter =0
        cv.imshow('camera', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()