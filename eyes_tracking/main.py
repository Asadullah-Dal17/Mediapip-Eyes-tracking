
import cv2 as cv 
import mediapipe as mp
import csv 
font = cv.FONT_HERSHEY_COMPLEX
file_path = 'selected_landmarks.csv'
with open(file_path, 'r') as csv_file:
    data_list= list(csv.reader(csv_file))
    LEFT_EYE =[int(i) for i in data_list[1][1:]]
    RIGHT_EYE = [int(i) for i in data_list[0][1:]]
    print(RIGHT_EYE)
mp_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture(0)
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

def blink_ratio(frame,landmarks, eye_points):
    # horizontal line data 
    hx, hy = landmarks[eye_points[0]][1]
    
    hx1, hy1 = landmarks[eye_points[7]][1]
    # Top_Vertical Points 
    second_x,vy11 = landmarks[eye_points[12]][1]
    first_x,vy = landmarks[eye_points[11]][1]
    padding = int((first_x -second_x)/2)
    # bottom Vertical Point 
    vx1, vy1 = landmarks[eye_points[4]][1]
    # print(padding)
    # cv.circle(frame, (first_x+padding, vy ), 4,(0,255,255), -1) 
    # cv.circle(frame, (hx,hy ), 2,(0,255,255), -1)
    # cv.circle(frame, (hx1,hy1 ), 2,(0,255,255), -1)
    # cv.circle(frame, (second_x,vy11 ), 2,(0,255,255), -1)
    cv.line(frame, (hx, hy), (hx1, hy1), (255,0,0),2)
    cv.line(frame, (vx1, vy1), (second_x+padding,vy ), (0,255,255))
    eye_pixel_with = hx1-hx 
    # print(h_px)
    eye_pixel_height = vy1 -vy
    ratio = eye_pixel_with/ eye_pixel_height
    # print(ratio)



    return ratio
    # print('x , y ', x,' , ',y , 'x1 , y1 ', x1, ' , ',y1)
    
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = camera.read()
        
        if ret is False:
            break
        frame = cv.resize(frame, None, fx=1.8, fy=1.8, interpolation=cv.INTER_LINEAR_EXACT)
        # converting color space from BGR to RGB 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # getting the frame width 
        height, width = frame.shape[:2]
        # print(width, height)
        rgb_frame.flags.writeable=False
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            image, points = landmarks_detector(rgb_frame, results)
            # print(len(LEFT_EYE))
            
            # [cv.circle(frame, (points[pos][1]), 1,(0,0,255), -1) for pos in LEFT_EYE]
            # [cv.circle(frame, (points[pos][1]), 1,(255,0,255), -1) for pos in RIGHT_EYE]
            
            # right_h=points[RIGHT_EYE[0]][1]
            # right_h2=points[RIGHT_EYE[7]][1]
            
            top_point =points[LEFT_EYE[4]][1]
            bottom_point=points[LEFT_EYE[11]][1]
            lfx=points[LEFT_EYE[11]][1][0]
            first_x, first_y = points[LEFT_EYE[11]][1]
            # cv.circle(frame, (points[LEFT_EYE[11]][1]), 2,(255,255,0), -1)
            # cv.circle(frame, (points[LEFT_EYE[12]][1]), 2,(0,255,0), -1)
            ratio=blink_ratio(frame ,points, LEFT_EYE)
            second_x = points[LEFT_EYE[12]][1][0]
            pad_to_center = int((first_x -second_x)/2)
            cv.circle(frame, (second_x+pad_to_center,first_y ), 1,(255,0,255), -1) 
            # print(" te ",pad_to_center)
            lf_x = points[LEFT_EYE[0]][1]
            lf_x1 = points[LEFT_EYE[7]][1][0]

            y = top_point[1]
            y1 = bottom_point[1]

            dif =(lf_x1-lf_x[0])/(y-y1)
            if ratio>5:
                print("blink")
                cv.putText(frame, "BLINK :)", (70, 70), font, 0.7, (0, 255,0), 2)

            # print(f'y: {y} - y1 {y1} = {y-y1} || x: {lf_x[0]} - y1 {lf_x1} = {lf_x1-lf_x[0]} :ratio : {dif}' )
            # cv.line(frame, top_point, (second_x+pad_to_center,first_y ), (255,0,0))

        cv.imshow('camera', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
    cv.destroyAllWindows()
    camera.release()