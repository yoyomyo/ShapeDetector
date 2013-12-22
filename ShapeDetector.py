__author__ = 'yoyomyo'

import numpy as np
import cv2
import os,sys
import svgwrite
from svgwrite import px
import pdb

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
    if w > 1000 or h > 1000:
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
    img3 = cv2.dilate(img2,element1)
    # use the complement of the dilated image
    img3 = 255-img3
    return img,img3

# parameter used in this function:
# area threshold: 100
def get_training_samples(color_img, bw_img):
    samples =  np.empty((0,3))
    responses = []
    keys = [48,49] # key responses are 0 or 1

    contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),2)

            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            cv2.drawContours(color_img,[approx],-1,(0,255,0),1)

            extent = get_extent(cnt);
            solidity = get_solidity(cnt);
            extent_poly = get_extent_poly(cnt)

            cv2.imshow('norm',color_img)
            key = cv2.waitKey(0)
            if key == 27:
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = np.array([[extent_poly, extent, solidity]])
                samples = np.append(samples,sample,0)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "training complete"

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
    cv2.destroyAllWindows()

def get_classifier():
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    model = cv2.SVM()
    model.train(samples,responses)
    return model

# should return a list of rectangles
def get_rectangles(color_img, bw_img, svm):
    contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    height,width = bw_img.shape
    result = []
    #should filter out the contour that has only one child and is too close to that child
    # print hierarchy.shape
    # print hierarchy[:,0,:]
    # print contours[0].shape
    idx = 0
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:

            extent = get_extent(cnt)
            solidity = get_solidity(cnt)
            extent_poly = get_extent_poly(cnt)
            sample = np.array([[extent_poly, extent, solidity]], np.float32)
            is_rect = svm.predict(sample)
            if is_rect:
                rectangles.append(idx)
                # [x,y,w,h] = cv2.boundingRect(cnt)
                # cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),1)
                # cv2.drawContours(color_img,[cnt],-1,(0,255,0),1)
                # cv2.drawContours(color_img,[cnt],-1,(0,255,0),1)
                # approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                # hull = cv2.convexHull(cnt)
        idx += 1

    for idx in rectangles:
        cnt = contours[idx]
        hier = hierarchy[:,idx,:]
        first_child = hier[0][3]
        child_next = hierarchy[:,first_child,:][0][0]
        child_prev = hierarchy[:,first_child,:][0][1]
        has_one_child = first_child in rectangles and\
                     child_next not in rectangles and\
                     child_prev not in rectangles

        child_cnt = contours[first_child]
        parent_area = cv2.contourArea(cnt)
        child_area = cv2.contourArea(child_cnt)

        if has_one_child and (child_area/parent_area > 0.8):
            pass
        else:
            [x,y,w,h] = cv2.boundingRect(cnt)
            to_add = True
            # for box in result:
            #     if is_nested_bounding_box(box, [x,y,w,h], width, height):
            #         to_add = False
            if to_add:
                result.append((x,y,w,h))
            cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),1)

    show_image_in_window('contour', color_img)
    result = [(height, width)]+result
    return result

def is_nested_bounding_box(box1, box2, width, height):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    w = 10
    h = 10
    return abs(x1-x2)<10 and abs(y1-y2)<10 and abs(w1-w2) < w and abs(h1-h2) < h

def get_contour_children(hierarchy):
    return []

def get_extreme_points(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    return [leftmost, rightmost, topmost, bottommost]

def get_extent_poly(cnt):
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    area = cv2.contourArea(approx)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent

def get_extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent

def get_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity

def resize_image(im, new_shape):
    blur = cv2.GaussianBlur(im,(5,5),0)
    small = cv2.resize(blur,new_shape)
    return small

def run_shape_detector(path_to_img, training=False):
    im = cv2.imread(path_to_img)
    color, bw = preprocess_image(im)
    training = False
    if training:
        get_training_samples(color, bw)
    else:
        svm = get_classifier()
        result = get_rectangles(color,bw,svm)
        return result

def generate_svg(list_of_rect, filename):
    height, width = list_of_rect.pop(0)

    dwg = svgwrite.Drawing(filename, size = ((width+6)*px, (height+6)*px))
    dwg.add(dwg.rect(insert=(3*px, 3*px), size=(width*px, height*px), fill='white', stroke='black', stroke_width=3))
    for x,y,w,h in list_of_rect:
        rect = dwg.add(dwg.rect(insert=(x*px, y*px), size=(w*px, h*px)))
        rect.fill('white',opacity=0.5).stroke('black', width=3)
    dwg.save()

# result = run_shape_detector('sketches/sketch4.png', training=False)
# generate_svg(result, 'test.svg')
#
for root, subdirs, files in os.walk('sketches'):
    count = 0
    for file in files:
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
            path_to_file =  os.path.join(root,file)
            result = run_shape_detector(path_to_file, training=False)
            generate_svg(result,'result%d.svg'%count)
            #cv2.imwrite('result%d.png'%count,result)
            count += 1

# print result