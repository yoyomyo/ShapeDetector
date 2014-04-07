__author__ = 'yoyomyo'
import os,sys,math,pdb
import numpy as np
import cv2
from svg_generator import *
from project_helpers import *
from Queue import deque

class ShapeDetector:

    CIRCLE,ELLIPSE,RECT,TRIANGLE, LINE = 0,1,2,3,4

    MAX_IMG_DIM = 1000              # if width or height greater than 1000 pixels, processing image would be slow
    MORPH_DIM = (3,3)
    CONTOUR_THRESHOLD = 50          # if a contour does not contain more than 50 pixels,
                                    # then we are not interested in the contour
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    YELLOW = (0,255,255)

    EPS = 0.0001                    # used to prevent the division by zero or taking lo

    NUM_CLASS = 3

    NESTED_CONTOUR_DISTANCE = 15    #if a two contours are nested and their boundaries
                                    # are within 10px then remove the innner conotur
    PT_DISTANCE_THRESHOLD = 20      # this distance may change, threshold should depend on how big the contour is

    ALIGNMENT_DRIFT_THRESHOLD = 0.1

    def get_data(self, dir):
        i = 0
        for root, subdirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    path_to_img = os.path.join(root,file)
                    print path_to_img
                    img = cv2.imread(path_to_img)
                    shapes = self.get_shapes_from_img(img, i)
                    h,w,c = img.shape
                    i += 1
                    self.generate_svg(shapes, 'test'+ str(i) +'.svg', w, h)

    # get shape features from img
    def get_shapes_from_img(self, contours, img, num):
        color, bw_img = preprocess_image(img)
        shapes = self.get_shapes(contours, color, bw_img)
        return shapes

    def get_shapes(self, contours, color_img, bw_img):
        #contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours =  filter_contours(contours)
        shapes = [] # a shape is a dictionary containing necessary information to draw the shape in SVG
        rectangles = [] # this will be used to detect tables
        for h, cnt in enumerate(contours):
            points = self.get_key_points(cnt)
            shape = self.infer_shape_from_key_points(points, cnt)

            # for pt in points:
            #     cv2.circle(color_img, tuple(pt[0]),3, self.RED, thickness=2, lineType=8, shift=0)
            #     #txt = "%.2f" % (d/math.pi*180)
            #     #cv2.putText(color_img, txt, tuple(current[0]+[4,4 ]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, self.RED, 1, 8)
            # show_image_in_window('c', color_img)

            if shape == self.CIRCLE:
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                # cv2.circle(color_img,center,radius,self.RED,2)
                shapes.append({
                    'shape':self.CIRCLE,
                    'center':(int(x),int(y)),
                    'radius':int(radius)
                })
            elif shape == self.ELLIPSE:
                ellipse = cv2.fitEllipse(cnt)
                # cv2.ellipse(color_img,ellipse,self.GREEN,2)
                shapes.append({
                    'shape':self.ELLIPSE,
                    'ellipse': ellipse
                })

            elif shape == self.RECT:
                # x,y,w,h = cv2.boundingRect(cnt)
                # cv2.rectangle(color_img, (x,y), (x+w,y+h), self.YELLOW, 2)
                rect = cv2.minAreaRect(cnt)
                box = cv2.cv.BoxPoints(rect)

                # make the box horizontal if the box is horizontal

                hor_box = is_horizontal_box(box)
                box = hor_box if hor_box else box
                box = np.int0(  (np.array(box)) )

                # find if this box is a form/table
                rectangles.append(box)


                # shapes.append({
                #     'shape':self.RECT,
                #     'points':[tuple(pt) for pt in box]+[tuple(box[0])],
                # })
                #cv2.drawContours(color_img,[box],0,self.YELLOW,2)

            elif shape == self.TRIANGLE:
                # pdb.set_trace()
                points = np.int0(points)
                # cv2.line(color_img, tuple(points[0][0]), tuple(points[1][0]), self.BLUE, 2)
                # cv2.line(color_img, tuple(points[1][0]), tuple(points[2][0]), self.BLUE, 2)
                # cv2.line(color_img, tuple(points[2][0]), tuple(points[0][0]), self.BLUE, 2)
                shapes.append({
                    'shape':self.TRIANGLE,
                    'points':[tuple(pt[0]) for pt in points] + [tuple(points[0][0])]
                })

        # convert shape parameter to svg on image
        lines, boxes = self.find_table_box(rectangles, bw_img)

        for line in lines:
            shapes.append({
                'shape':self.LINE,
                'points':line
            })
        for box in boxes:
            shapes.append({
                'shape':self.RECT,
                'points':[tuple(pt) for pt in box]+[tuple(box[0])],
            })

        return shapes

    def get_key_points(self, cnt):
        previous_gradient = None
        diffs = []
        # mask = np.zeros(bw_img.shape,np.uint8)
        key_points = []
        for i,pt in enumerate(cnt):
            prev = cnt[(i-1)%len(cnt)]     # use neighbor points to compute gradient
            nxt = cnt[(i+1)%len(cnt)]

            gradient = self.get_gradient_angle(prev, nxt)

            if not previous_gradient:
                previous_gradient = gradient
            else:
                # compute gradient angle difference
                diff = math.fabs(previous_gradient-gradient)
                # if diff is greater than 320 degree, the gradient change is too big, set it to zero
                diff = 0 if diff > 300.0/180.0*math.pi else diff

                c = diff*255
                diffs.append(diff)
                # cv2.line(mask, tuple(prev[0]), tuple(nxt[0]), (c,c,c), thickness=1, lineType=8, shift=0)
                previous_gradient = gradient

        mmax = []
        # find the maximums
        for j, diff in enumerate(diffs):
            is_max = True
            for n in range(-3,4): # find the max in [j-2, j+2] neighborhood
                if n == 0: continue
                is_max = is_max and diff >= diffs[(j+n)%len(diffs)]
            if is_max:
                # print diffs[max(0, j-4) : min(j+5, len(diffs)-1)]
                # compute the gradient diffs of the max points again and find the inflection point
                mmax.append(j)
                # cv2.circle(mask, tuple(cnt[j+1][0]),3, (255,255,255), thickness=2, lineType=8, shift=0)
        # self.show_image_in_window('m', mask)

        # compute the gradient between the max
        for i, max_index in enumerate(mmax):
            prev_max_index = mmax[(i-1)%len(mmax)]
            next_max_index = mmax[(i+1)%len(mmax)]
            prev_max = cnt[prev_max_index]
            next_max = cnt[next_max_index]
            current = cnt[max_index]

            d = math.fabs(self.get_gradient_angle(prev_max, current) - self.get_gradient_angle(current, next_max))
            d = 0 if d > 300.0/180.0*math.pi else d

            if d > 0.25*math.pi:
                if any(dist(pt,current)<self.PT_DISTANCE_THRESHOLD for pt in key_points):
                    pass
                else:
                    key_points.append(current)

        return key_points

    def infer_shape_from_key_points(self, key_points, cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        if len(key_points) < 3:
            if abs((w+0.0)/(h+0.0)-1) < 0.3:
                return self.CIRCLE
            else:
                return self.ELLIPSE
        elif len(key_points) == 3:
            return self.TRIANGLE
        elif len(key_points) > 3:
            return self.RECT


    def find_table_box(self, rectangles, img):
        h,w = img.shape
        table = {}
        leftover_boxes =[]
        lines = []

        # this is to help find the locations of horizontal and verticle lines
        hor = []
        ver = []
        for i,box in enumerate(rectangles):
            table[i] = []

        for i,box1 in enumerate(rectangles):
            for j,box2 in enumerate(rectangles):
                if j<=i: continue
                if overlap_boxes(box1, box2):
                    table[i].append(box2)
                    table[j].append(box1)

        for k in table:
            box = rectangles[k]
            if len(table[k])>=2:
                for pt in box:
                    ver.append(pt[0])
                    hor.append(pt[1])
            else:
                leftover_boxes.append(box)


        hor = np.array(hor)
        ver = np.array(ver)
        hor_bin = [i for i in range(0,h,30)]
        ver_bin = [i for i in range(0,w,30)]
        hor_hist, _ = np.histogram(hor, bins=hor_bin)
        ver_hist, _ = np.histogram(ver, bins=ver_bin)

        hor = [ (hor_bin[i] + hor_bin[i+1])/2 for i,x in enumerate(hor_hist) if x > 2 ]
        ver = [ (ver_bin[i] + ver_bin[i+1])/2 for i,x in enumerate(ver_hist) if x > 2 ]

        for y in ver:
            lines.append(  [(y,hor[0]), (y,hor[-1])] )
        for x in hor:
            lines.append( [(ver[0],x), (ver[-1],x)] )

        return lines, leftover_boxes


    def get_gradient_angle(self, prev, nxt):
        tangent = nxt - prev
        dx,dy = tangent[0]
        a1 = math.atan2(dy,dx)
        if a1 < 0:
            return a1 + 2*math.pi
        else:
            return a1

    # given a list of shapes, draw the shape in svg
    # and save the svg to a file in the end
    def generate_svg(self, shapes, filename, width, height):
         gen = SVGGenerator(shapes)
         gen.generate_svg(filename, width, height)


# sd = ShapeDetector()
# sd.get_data('test')
