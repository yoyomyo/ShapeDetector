__author__ = 'yoyomyo'

import numpy as np
import cv2


def show_image_in_window(win_name, img):
    cv2.imshow(win_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# the parameters involved in this function are:
# width or height threshold 1000
# adaptiveThreshold BlockSize 21 and Constant 10
# erosion cross neighborhood (4,4)
# dilation cross neighborhood (3,3)
def preprocess_image(img):
    w,h,c = img.shape
    while w > 1000 and h > 1000:
        img = cv2.pyrDown(img)
        w,h,c = img.shape
    # convert image to grayscale
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv2.THRESH_BINARY,21, 10)
    ret,thresh = cv2.threshold(thresh,127,255,0)
    # apply erosion and dilation, this is for trying to close gaps in a contour
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    img2 = cv2.erode(thresh,element2)
    #show_image_in_window('eroded', img2)
    img3 = cv2.dilate(img2,element1)
    # use the complement of the dilated image
    img3 = 255-img3
    #show_image_in_window('dilated', img3)
    return img,img3

# parameter used in this function:
# area threshold: 100
def get_contours(color_img, bw_img):
    contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(color_img,[cnt],-1,(0,255,0),1)
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            hull = cv2.convexHull(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.drawContours(color_img,[approx],-1,(0,255,0),1)

    show_image_in_window('contour', color_img)
    return contours, hierarchy

im = cv2.imread('sketches/sketch1.png')
color, bw = preprocess_image(im)
contours, hierarchy = get_contours(color, bw)

