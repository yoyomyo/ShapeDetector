__author__ = 'yoyomyo'

import numpy as np
import cv2
from shape_detector import *
from text_detector import *
from svg_generator import *

class Project:

    MAX_IMG_DIM = 1000              # if width or height greater than 1000 pixels, processing image would be slow
    MORPH_DIM = (3,3)
    CONTOUR_THRESHOLD = 50          # if a contour does not contain more than 50 pixels,
                                    # then we are not interested in the contour

    NESTED_CONTOUR_DISTANCE = 10    #if a two contours are nested and their boundaries
                                    # are within 10px then remove the innner conotur

    def __init__(self):
        self.text_detector = TextDetector()
        self.shape_detector = ShapeDetector()
        self.svg_generator = SVGGenerator()


    # get a list of contours from a dir
    def get_data(self, dir):
        i = 0
        for root, subdirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    path_to_img = os.path.join(root,file)
                    # print path_to_img, img_class
                    # pdb.set_trace()
                    self.get_contours_from_img(path_to_img, i)
                    i += 1

    # get contours from img
    def get_contours_from_img(self, path_to_img, num):
        img = cv2.imread(path_to_img)
        color, bw_img = self.preprocess_image(img)
        contours = self.get_contours(bw_img)

    def preprocess_image(self, img):
        w,h,c = img.shape
        while w > self.MAX_IMG_DIM or h > self.MAX_IMG_DIM:
            img = cv2.pyrDown(img)
            w,h,c = img.shape
        # convert image to grayscale
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # generate adaptive thresholding parameters
        thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        cv2.THRESH_BINARY,21, 10)
        ret,thresh = cv2.threshold(thresh,127,255,0)
        # apply erosion and dilation, this is for trying to close gaps in a contour
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT,self.MORPH_DIM)
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT,self.MORPH_DIM)
        img2 = cv2.erode(thresh,element2)
        img3 = cv2.dilate(img2,element1)
        # use the complement of the dilated image
        img3 = 255-img3
        #self.show_image_in_window('preprocess', img3)
        return img,img3


    def get_contours(self, bw_img):
        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = self.filter_contours(contours)
        return contours

    def filter_contours(self, contours):
        contours = filter(lambda cnt: len(cnt) > self.CONTOUR_THRESHOLD, contours)
        contours = self.remove_nested_contours(contours)
        return contours

    def remove_nested_contours(self, contours):
        contour_table = {}
        for i, cnt in enumerate(contours):
            to_add = self.get_contour_extreme_points(cnt)
            if len(contour_table) == 0:
                contour_table[i] = to_add
            else:
                to_replace = i
                for k in contour_table:
                    flag = self.are_nested_contour(contour_table[k], to_add)
                    if flag == 1:
                        # former contour is larger, do not add
                        to_replace = -1
                    elif flag == 2:
                        # latter contour is larger, replace
                        to_replace = k

                if to_replace != -1:
                    contour_table[to_replace] = to_add

        indexes = contour_table.keys()
        return [contours[i] for i in indexes]

    # return 1 if extreme_points2 is nested within extreme_points1
    # return 2 if extreme_points1 is nested within extreme_points2
    # otherwise return 3
    def are_nested_contour(self, extreme_points1, extreme_points2):
        left1, right1, top1, bottom1 = extreme_points1
        left2, right2, top2, bottom2 = extreme_points2
        xl1, yl1 = left1
        xl2, yl2 = left2
        xr1, yr1 = right1
        xr2, yr2 = right2
        xt1, yt1 = top1
        xt2, yt2 = top2
        xb1, yb1 = bottom1
        xb2, yb2 = bottom2

        a = self.within_contour_distance(xl1, xl2)
        b = self.within_contour_distance(xr2, xr1)
        c = self.within_contour_distance(yt1, yt2)
        d = self.within_contour_distance(yb2, yb1)

        e = self.within_contour_distance(xl2, xl1)
        f = self.within_contour_distance(xr1, xr2)
        g = self.within_contour_distance(yt2, yt1)
        h = self.within_contour_distance(yb1, yb2)

        if e and f and g and h:
            return 1
        elif a and b and c and d:
            return 2
        return 3

    def within_contour_distance(self, p1, p2):
        return within_distance(p1,p2,self.NESTED_CONTOUR_DISTANCE)
