__author__ = 'yoyomyo'

import numpy as np
import cv2
import sys, os
import os.path
from project_helpers import *

# This is a class utilizes a linear SVM classifier
# and inner distance to distinguish text regions
# and shape regions in a sketch

class TextDetector:

    def __init__(self):
        self.svm = None

    def train(self, dir):
        if os.path.isfile("feat.data"): os.remove("feat.data")
        if os.path.isfile("tag.data"):os.remove("tag.data")

        for c in [0,1]:
            subdir = os.path.join(dir,str(c))
            for root, subdirs, files in os.walk(subdir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                        path_to_img = os.path.join(root,file)
                        img = cv2.imread(path_to_img)
                        color, bw_img = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)
                        self.get_training_samples(color, bw_img, c)

        self.svm = self.get_classifier()

    def test(self, dir):
        i = 1
        for root, subdirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    path_to_img = os.path.join(root,file)
                    img = cv2.imread(path_to_img)
                    color, bw_img = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)
                    self.get_testing_result(color, bw_img, i)
                    i += 1

    def get_testing_result(self, color_img, bw_img, index):
        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # should filter out the contour that has only one child and is too close to that child
        # print hierarchy.shape
        # print hierarchy[:,0,:]
        # print contours[0].shape
        idx = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                feature = self.get_features(cnt)
                feature = np.array([feature], np.float32)
                is_shape = self.svm.predict(feature)
                [x,y,w,h] = cv2.boundingRect(cnt)
                if is_shape:
                    cv2.rectangle(color_img,(x,y),(x+w,y+h),BLUE,1)
                    # cv2.drawContours(color_img,[cnt],-1,(0,255,0),1)
                    # cv2.drawContours(color_img,[cnt],-1,(0,255,0),1)
                    # approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                    # hull = cv2.convexHull(cnt)
                else:
                   cv2.rectangle(color_img,(x,y),(x+w,y+h),GREEN,1)
            idx += 1
        show_image_in_window('color', color_img)
    # parameter used in this function:
    # area threshold: 100
    def get_training_samples(self,color_img, bw_img, c):
        features =  np.empty((0,2))
        tags = []
        # keys = [48,49] # key responses are 0 or 1

        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(color_img,(x,y),(x+w,y+h),RED,2)

                # approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                # cv2 .drawContours(color_img,[approx],-1,(0,255,0),1)
                feature = self.get_features(cnt)

                #cv2.imshow('norm',color_img)
                # key = cv2.waitKey(0)
                # key = 49
                # if key == 27:
                #     sys.exit()
                # elif key in keys:
                tags.append(c)
                sample = np.array([feature])
                features = np.append(features,sample,0)

        tags = np.array(tags,np.float32)
        tags = tags.reshape((tags.size,1))

        np.savetxt('tmp_feat.data',features)
        np.savetxt('tmp_tag.data',tags)

        # append temp data to training data
        append_result_to_file("feat.data", "tmp_feat.data")
        append_result_to_file("tag.data", "tmp_tag.data")

        cv2.destroyAllWindows()


    def get_features(self, cnt):
        # aspect ratio w/h
        # compactness sqrt(area)/perimeter
        # number of holes
        # horizontal crossing
        # convex hull ratio AREAconvex/AREA

        x,y,w,h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)

        aspect = float(w)/h
        compactness = float(w+h) / perimeter
        solidity = get_solidity(cnt)
        convex_hull_ratio = float(area)/w*h
        return [solidity, area]

    def get_classifier(self):
        samples = np.loadtxt('feat.data',np.float32)
        responses = np.loadtxt('tag.data',np.float32)
        responses = responses.reshape((responses.size,1))
        model = cv2.SVM()
        model.train(samples,responses)
        return model

def preprocess_image(img, max_img_dim, morph_dim):
    w,h,c = img.shape
    while w > max_img_dim or h > max_img_dim:
        img = cv2.pyrDown(img)
        w,h,c = img.shape
    # convert image to grayscale
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # generate adaptive thresholding parameters
    thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,21, 10)
    ret,thresh = cv2.threshold(thresh,127,255,0)
    # apply erosion and dilation, this is for trying to close gaps in a contour
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT,morph_dim)
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT,morph_dim)
    img2 = cv2.erode(thresh,element2)
    img3 = cv2.dilate(img2,element1)
    # use the complement of the dilated image
    img3 = 255-img3
    # show_image_in_window('preprocess', img3)
    # cv2.imshow('black ',img3)
    return img,img3

def get_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity

# append file2 to file1
def append_result_to_file(file1, file2):
    with open(file1, "a") as f1:
        with open(file2, "r") as f2:
            f1.write(f2.read())

rd = TextDetector()
rd.train('train/')
rd.test('test')