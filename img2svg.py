__author__ = 'yoyomyo'

import numpy as np
import cv2
import sys
from shape_detector import *
from text_detector import *
from svg_generator import *

class img2svg:

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

    def get_texts_and_shapes(self, path_to_img, index, write_intermediate_result):
        img = cv2.imread(path_to_img)
        color, bw = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)

        w,h = bw.shape
        contours,hierarchy = cv2.findContours(bw,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        text_regions = self.text_detector.get_texts(contours, color, bw)

        for region in text_regions:
            cv2.rectangle(color,(region.left,region.top),(region.right, region.bottom),GREEN,1)

        _, bw = preprocess_image(img, MAX_IMG_DIM, MORPH_DIM)
        shapes, color = self.shape_detector.get_shapes(contours, color, bw, text_regions)

        if write_intermediate_result:
            cv2.imwrite("result" + str(index) + ".jpg", color)

        self.generate_svg(shapes, text_regions, "result" + str(index) + ".svg", h,w)

    # given a list of shapes, draw the shape in svg
    # and save the svg to a file in the end
    def generate_svg(self, shapes, texts, filename, width, height):
        gen = SVGGenerator(shapes, texts)
        gen.generate_svg(filename, width, height)

    def run(self, dir_name, write_intermediate_result=False):
        if dir_name:
            i = 1
            for root, subdirs, files in os.walk(dir_name):
                for file in files:
                    if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                        path_to_img = os.path.join(root,file)
                        print path_to_img
                        project.get_texts_and_shapes(path_to_img, i, write_intermediate_result)
                        i += 1


if __name__ == '__main__':
    project = img2svg()
    if len(sys.argv) == 2:
        project.run(sys.argv[1], bool(sys.argv[2]))
    else:
        project.run(sys.argv[1])


