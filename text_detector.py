__author__ = 'yoyomyo'

import numpy as np
import cv2
import sys, os
from project_helpers import *
from bisect import *

import pdb

TEXT_AREA_THRESHOLD_LOWER = 10
TEXT_HORIZONTAL_DISTANCE_THRESHOLD = 50
TEXT_VERTICAL_DISTANCE_THRESHOLD = 30
TEXT_PERCENT_THRESHOLD = 0.5

# This is a class utilizes a linear SVM classifier
# to distinguish text regions
# and shape regions in a sketch
# output a list of text regions
class TextDetector:

    def __init__(self):
        self.train('train/')

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
                    regions = self.get_texts(color, bw_img)
                    for region in regions:
                        if region.is_text_region():
                            cv2.rectangle(color,(region.left,region.top),(region.right,region.bottom),GREEN,1)
                        else:
                            cv2.rectangle(color,(region.left,region.top),(region.right,region.bottom),BLUE,1)

                    show_image_in_window('color', color)
                    cv2.imwrite("result" + str(i) + ".jpg", color)
                    i += 1

    def get_texts(self, contours, color_img, bw_img):
        contours2, is_text_flags = self.get_testing_result(contours, color_img, bw_img)
        contour_tables, text_regions = self.get_text_regions(contours2, is_text_flags)
        return contours2,  contour_tables, text_regions

    def get_testing_result(self, contours, color_img, bw_img):
        width, height = bw_img.shape
        #contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        #idx = 0    #just for writing to file needs

        is_text_flags = []

        # used for sorting
        contour_left = []
        filtered_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > TEXT_AREA_THRESHOLD_LOWER:
                x,y,w,h = cv2.boundingRect(cnt)

                #sort contours by their left bound
                i = bisect(contour_left, x)
                contour_left.insert(i, x)
                filtered_contours.insert(i, cnt)

                feature1, feature2, feature3 = self.get_features(cnt, width, height)

                feature1 = np.array([feature1], np.float32)
                feature2 = np.array([feature2], np.float32)
                feature3 = np.array([feature3], np.float32)

                is_shape1 = self.svm1.predict(feature1)
                is_shape2 = self.svm2.predict(feature2)
                is_shape3 = self.svm3.predict(feature3)

                is_text = not (is_shape1 and is_shape2 and is_shape3)
                is_text_flags.append(is_text)

        return filtered_contours, is_text_flags

    def get_training_samples(self,color_img, bw_img, c):
        width, height = bw_img.shape
        features1 = np.empty((0,1))
        features2 = np.empty((0,1))
        features3 = np.empty((0,1))
        tags = []
        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(color_img,(x,y),(x+w,y+h),RED,2)

                # approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                # cv2.drawContours(color_img,[approx],-1,(0,255,0),1)
                feature1, feature2, feature3 = self.get_features(cnt, width, height)

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

        #show_image_in_window('contour', color_img)
        cv2.destroyAllWindows()

    # the features must has something to do with the context too
    def get_features(self, cnt, width, height):
        # aspect ratio w/h
        # compactness sqrt(area)/perimeter
        # number of holes
        # horizontal crossing
        # convex hull ratio AREAconvex/AREA

        x,y,w,h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

        area = cv2.contourArea(cnt)
        # perimeter = cv2.arcLength(cnt,True)
        #
        # aspect = float(w)/h
        # compactness = float(w+h) / perimeter
        solidity = get_solidity(cnt)
        convex_hull_ratio = float(area)/w*h

        return solidity, convex_hull_ratio, area

    def get_classifier(self, feat, tag):
        samples = np.loadtxt(feat,np.float32)
        responses = np.loadtxt(tag,np.float32)
        responses = responses.reshape((responses.size,1))
        model = cv2.SVM()
        model.train(samples,responses)
        return model

    def get_text_regions(self, contours, is_text_flags):
        regions = []
        contour_tables = {}
        for i, cnt in enumerate(contours):
            is_text = is_text_flags[i]
            # merge this cnt into the list if possible
            merged = False
            for j, region in enumerate(regions):
                if region.merge(cnt, is_text):
                    merged = True
                    contour_tables[i] = j
                    break

            if not merged:
                image_region = ImageRegion(cnt, is_text)
                contour_tables[i] = len(regions)
                regions.append(image_region)

        return contour_tables, regions


# an ImageRegion is either text or shape
class ImageRegion:

    text_cnt_count = 0
    def __init__(self, cnt, is_text):
        self.list_of_contour = [cnt]
        x,y,w,h = cv2.boundingRect(cnt)
        # region boundary
        self.top = y
        self.bottom = y+h
        self.left = x
        self.right = x+w
        # keep track of how many text contours are in the region
        self.text_cnt_count += 1 if is_text else 0

    # updates list_of_contour
    # updates bounds
    def merge(self, contour, is_text):
        x,y,w,h = cv2.boundingRect(contour)
        if self.isAdjacent(contour):
            self.list_of_contour.append(contour)

            self.left = min(x, self.left)
            self.right = max(x+w, self.right)
            self.top = min(y, self.top)
            self.bottom = max(y+h, self.bottom)

            self.text_cnt_count += 1 if is_text else 0
            return True
        return False

    # tell if a contour should be merge with the region
    def isAdjacent(self, cnt):
        x,y,w,h = cv2.boundingRect(cnt)

        left = abs(self.left - (x+w)) < TEXT_HORIZONTAL_DISTANCE_THRESHOLD
        right = abs(x - self.right) < TEXT_HORIZONTAL_DISTANCE_THRESHOLD

        top = abs(self.top - y) < TEXT_HORIZONTAL_DISTANCE_THRESHOLD
        bottom = abs(self.bottom - (y+h)) < TEXT_HORIZONTAL_DISTANCE_THRESHOLD

        return (left or right) and (top and bottom)

    def is_text_region(self):
        return len(self.list_of_contour) > 1 and \
               self.text_cnt_count / float(len(self.list_of_contour)) >= TEXT_PERCENT_THRESHOLD and \
               abs(self.top - self.bottom) < 80

# rd = TextDetector()
# rd.train('train/')
# rd.test('test')