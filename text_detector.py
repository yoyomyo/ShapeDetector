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

    TEXT_AREA_THRESHOLD = 10

    def __init__(self):
        self.svm = None

    def train(self, dir):
        if os.path.isfile("feat1.data"): os.remove("feat1.data")
        if os.path.isfile("tag1.data"):os.remove("tag1.data")

        if os.path.isfile("feat2.data"): os.remove("feat2.data")
        if os.path.isfile("tag2.data"):os.remove("tag2.data")

        if os.path.isfile("feat3.data"): os.remove("feat3.data")
        if os.path.isfile("tag3.data"):os.remove("tag3.data")

        for c in [0,1]:
            subdir = os.path.join(dir,str(c))
            for root, subdirs, files in os.walk(subdir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                        path_to_img = os.path.join(root,file)
                        img = cv2.imread(path_to_img)
                        color, bw_img = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)
                        self.get_training_samples(color, bw_img, c)

        self.svm1 = self.get_classifier('feat1.data', 'tag1.data')
        self.svm2 = self.get_classifier('feat2.data', 'tag2.data')
        self.svm3 = self.get_classifier('feat3.data', 'tag3.data')

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

        idx = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.TEXT_AREA_THRESHOLD:
                feature1, feature2, feature3 = self.get_features(cnt)

                feature1 = np.array([feature1], np.float32)
                feature2 = np.array([feature2], np.float32)
                feature3 = np.array([feature3], np.float32)

                is_shape1 = self.svm1.predict(feature1)
                is_shape2 = self.svm2.predict(feature2)
                is_shape3 = self.svm3.predict(feature3)

                [x,y,w,h] = cv2.boundingRect(cnt)
                if is_shape2 and is_shape3 and is_shape1:
                    cv2.rectangle(color_img,(x,y),(x+w,y+h),BLUE,1)
                else:
                    cv2.rectangle(color_img,(x,y),(x+w,y+h),GREEN,1)
                    # merge the contours into one box

            idx += 1
        #show_image_in_window('color', color_img)
        cv2.imwrite("result" + str(index) + ".jpg", color_img)


    def get_training_samples(self,color_img, bw_img, c):
        features1 =  np.empty((0,1))
        features2 =  np.empty((0,1))
        features3 =  np.empty((0,1))
        tags = []
        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(color_img,(x,y),(x+w,y+h),RED,2)

                # approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                # cv2.drawContours(color_img,[approx],-1,(0,255,0),1)
                feature1, feature2, feature3 = self.get_features(cnt)

                tags.append(c)
                sample1 = np.array([[feature1]])
                sample2 = np.array([[feature2]])
                sample3 = np.array([[feature3]])

                features1 = np.append(features1,sample1,0)
                features2 = np.append(features2,sample2,0)
                features3 = np.append(features3,sample3,0)

        tags = np.array(tags,np.float32)
        tags = tags.reshape((tags.size,1))

        np.savetxt('tmp_feat1.data',features1)
        np.savetxt('tmp_feat2.data',features2)
        np.savetxt('tmp_feat3.data',features3)
        np.savetxt('tmp_tag.data',tags)

        # append temp data to training data
        append_result_to_file("feat1.data", "tmp_feat1.data")
        append_result_to_file("feat2.data", "tmp_feat2.data")
        append_result_to_file("feat3.data", "tmp_feat3.data")

        append_result_to_file("tag1.data", "tmp_tag.data")
        append_result_to_file("tag2.data", "tmp_tag.data")
        append_result_to_file("tag3.data", "tmp_tag.data")

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
        return solidity, convex_hull_ratio, area,

    def get_classifier(self, feat, tag):
        samples = np.loadtxt(feat,np.float32)
        responses = np.loadtxt(tag,np.float32)
        responses = responses.reshape((responses.size,1))
        model = cv2.SVM()
        model.train(samples,responses)
        return model

rd = TextDetector()
rd.train('train/')
rd.test('test')