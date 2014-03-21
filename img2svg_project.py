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

    def get_contours_from_img(self, path_to_img, num):
        img = cv2.imread(path_to_img)
        color, bw_img = preprocess_image(img)
        contours = self.get_contours(bw_img)

    def get_contours(self, bw_img):
        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = filter_contours(contours)
        return contours
