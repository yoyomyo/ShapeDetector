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
        self.svg_generator = None

    def get_texts_and_shapes(self, path_to_img, index):
        img = cv2.imread(path_to_img)
        color, bw = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)
        # show_image_in_window('bw', bw)
        w,h = bw.shape
        contours,hierarchy = cv2.findContours(bw,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        contours2, contour_tables, text_regions = self.text_detector.get_texts(contours, color, bw)

        non_text_contours = []
        for i, cnt in enumerate(contours2):
            if not text_regions[contour_tables[i]].is_text_region():
                non_text_contours.append(cnt)

        color, bw = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)
        shapes = self.shape_detector.get_shapes(non_text_contours, color, bw)


        texts = []
        for region in text_regions:
            if region.is_text_region():
                texts.append(region)

        self.generate_svg(shapes, texts, "result" + str(index) + ".svg", h,w)


    # given a list of shapes, draw the shape in svg
    # and save the svg to a file in the end
    def generate_svg(self, shapes, texts, filename, width, height):
        gen = SVGGenerator(shapes, texts)
        gen.generate_svg(filename, width, height)

project = Project()
i = 1
for root, subdirs, files in os.walk('test'):
    for file in files:
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
            path_to_img = os.path.join(root,file)
            print path_to_img
            project.get_texts_and_shapes(path_to_img, i)
            i += 1
