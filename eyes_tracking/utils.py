'''

Author: Asadullah Dal 
Youtube Channel: https://www.youtube.com/c/aiphile

'''

import cv2 as cv 
import numpy as np

# colors 
# values =(blue, green, red) opencv accepts BGR values not RGB
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MEGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = ((147,20,255))

def drawColor(img, colors):
    x, y = 10,2
    w, h = 40, 40
    for color in colors:
        x += w+5 
        # y += 10 
        cv.rectangle(img, (x, y ), (x+w, y+h), color, -1)
    

def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background 
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    overlay = img.copy() # coping the image
    cv.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0) # overlaying the rectangle on the image.
    cv.putText(new_img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text
    img = new_img

    return img


def textBlurBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0),kneral=(33,33) , pad_x=3, pad_y=3):
    """    
    Draw text with background blured,  control the blur value, with kernal(odd, odd)
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param kneral: tuple(3,3) int as odd number:  higher the value, more blurry background would be
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels)  padding of in y direction
    @return: img mat, with text drawn, with background blured
    
    call the function: 
     img =textBlurBackground(img, 'Blured Background Text', cv2.FONT_HERSHEY_COMPLEX, 0.9, (20, 60),2, (0,255, 0), (49,49), 13, 13 )
    """
    
    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    blur_roi = img[y-pad_y-t_h: y+pad_y, x-pad_x:x+t_w+pad_x] # croping Text Background
    img[y-pad_y-t_h: y+pad_y, x-pad_x:x+t_w+pad_x]=cv.blur(blur_roi, kneral)  # merging the blured background to img
    cv.putText(img,text, textPos,font, fontScale, textColor,textThickness )          
    # cv.imshow('blur roi', blur_roi)
    # cv.imshow('blured', img)
    return img



def main():
    cap = cv.VideoCapture(0)
    while True:
        success, img = cap.read()
        drawColor(img, [BLACK,WHITE ,BLUE,RED,CYAN,YELLOW,MEGENTA,GRAY ,GREEN,PURPLE,ORANGE,PINK])
        textBlurBackground(img, 'Blured Background Text', cv.FONT_HERSHEY_COMPLEX, 0.8, (60, 140),2, GREEN, (71,71), 13, 13)
        img=textWithBackground(img, 'Colored Background Texts', cv.FONT_HERSHEY_COMPLEX, 0.6, (60,70), textThickness=2, bgColor=GREEN, textColor=BLACK, bgOpacity=0.7, pad_x=6, pad_y=6)
        # imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('img', img)
        if cv.waitKey(1) ==ord('q'):
            break
if __name__ == "__main__":
    main()
