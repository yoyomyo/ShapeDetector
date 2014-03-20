__author__ = 'yoyomyo'
import os,sys,math,pdb
import numpy as np
import cv2
from svg_generator import *

class ShapeDetector:

    CIRCLE,ELLIPSE,RECT,TRIANGLE = 0,1,2,3

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

    NESTED_CONTOUR_DISTANCE = 10    #if a two contours are nested and their boundaries
                                    # are within 10px then remove the innner conotur
    PT_DISTANCE_THRESHOLD = 20      # this distance may change, threshold should depend on how big the contour is

    ALIGNMENT_DRIFT_THRESHOLD = 5

    def get_data(self, dir):
        for root, subdirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    path_to_img = os.path.join(root,file)
                    # print path_to_img, img_class
                    # pdb.set_trace()
                    self.get_shapes_from_img(path_to_img)


    # get shape features from img, put them into ,
    # and store relevant shape into
    def get_shapes_from_img(self, path_to_img):
        img = cv2.imread(path_to_img)
        color, bw_img = self.preprocess_image(img)
        shapes = self.get_shapes(color, bw_img)
        h,w= bw_img.shape

        self.generate_svg(shapes, 'test.svg', w, h)

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

    def get_shapes(self, color_img, bw_img):
        contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = self.filter_contours(contours)
        shapes = [] # a shape is a dictionary containing necessary information to draw the shape in SVG
        for h, cnt in enumerate(contours):
            points = self.get_key_points(cnt)
            shape = self.infer_shape_from_key_points(points, cnt)


            # for pt in points:
            #     cv2.circle(color_img, tuple(pt[0]),3, self.RED, thickness=2, lineType=8, shift=0)
            #     #txt = "%.2f" % (d/math.pi*180)
            #     #cv2.putText(color_img, txt, tuple(current[0]+[4,4 ]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, self.RED, 1, 8)
            # self.show_image_in_window('c', color_img)

            if shape == self.CIRCLE:
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(color_img,center,radius,self.RED,2)
                shapes.append({
                    'shape':self.CIRCLE,
                    'center':(int(x),int(y)),
                    'radius':int(radius)
                })
            elif shape == self.ELLIPSE:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(color_img,ellipse,self.GREEN,2)
                shapes.append({
                    'shape':self.ELLIPSE,
                    'ellipse': ellipse
                })

            elif shape == self.RECT:
                # x,y,w,h = cv2.boundingRect(cnt)
                # cv2.rectangle(color_img, (x,y), (x+w,y+h), self.YELLOW, 2)
                rect = cv2.minAreaRect(cnt)
                box = cv2.cv.BoxPoints(rect)
                box = np.int0(box)
                # pdb.set_trace()
                shapes.append({
                    'shape':self.RECT,
                    'points':[tuple(pt) for pt in box]+[tuple(box[0])]
                })
                print [tuple(pt) for pt in box]+[tuple(box[0])]
                cv2.drawContours(color_img,[box],0,self.YELLOW,2)

            elif shape == self.TRIANGLE:
                # pdb.set_trace()
                points = np.int0(points)
                cv2.line(color_img, tuple(points[0][0]), tuple(points[1][0]), self.BLUE, 2)
                cv2.line(color_img, tuple(points[1][0]), tuple(points[2][0]), self.BLUE, 2)
                cv2.line(color_img, tuple(points[2][0]), tuple(points[0][0]), self.BLUE, 2)
                shapes.append({
                    'shape':self.TRIANGLE,
                    'points':[tuple(pt[0]) for pt in points] + [tuple(points[0][0])]
                })

        # convert shape parameter to svg on image
        self.show_image_in_window('c', color_img)
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


    def get_gradient_angle(self, prev, nxt):
        tangent = nxt - prev
        dx,dy = tangent[0]
        a1 = math.atan2(dy,dx)
        if a1 < 0:
            return a1 + 2*math.pi
        else:
            return a1

    def get_contour_centroid(self, cnt):
        # compute the centroid of the contour to help compute chord normals and orientation
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        return np.array([centroid_x,centroid_y])

    def get_contour_extreme_points(self, cnt):
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        return [leftmost, rightmost, topmost, bottommost]

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

    def is_horizontal_box(self, box):
        br, bl, tl, tr = box
        b_diff = abs(br[1]-bl[1])
        r_diff = abs(br[0]-tr[0])
        l_diff = abs(tl[0]-bl[0])
        t_diff = abs(tl[1]-tr[1])
        return all(x< 5 for x in [b_diff,r_diff,l_diff,t_diff])


    def get_extent(self, cnt):
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w*h
        extent = float(area)/rect_area
        return extent

    def show_image_in_window(self, win_name, img):
        cv2.imshow(win_name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # given a list of shapes, draw the shape in svg
    # and save the svg to a file in the end
    def generate_svg(self, shapes, filename, width, height):

         gen = SVGGenerator(shapes)

         gen.generate_svg(filename, width, height)


        # dwg = svgwrite.Drawing(filename, size = ((width+self.SVG_BOUNDARY)*px, (height+self.SVG_BOUNDARY)*px))
        #
        # dwg.add(dwg.rect(insert=(3*px, 3*px), size=(width*px, height*px), fill='white', stroke='black', stroke_width=3))
        # for entry in shapes:
        #     shape = entry['shape']
        #     if shape == self.CIRCLE:
        #         cx, cy = entry['center']
        #         r = entry['radius']
        #         circle = dwg.add(dwg.circle(center=(cx*px, cy*px),r=r*px))
        #         circle.fill('white',opacity=0.5).stroke('black', width=3)
        #     if shape == self.RECT:
        #         pts = entry['points']
        #         print pts
        #         rect = dwg.add(dwg.polyline(pts, stroke = 'black', fill='none'))
        #     elif shape == self.TRIANGLE:
        #         points = entry['points']
        #         print points
        #         triangle = dwg.add(dwg.polyline(points))
        #
        # dwg.save()


# global helper functions
def midpoint(p1, p2):
    return (p1+p2)/2

def normalize(v):
    return v/np.linalg.norm(v)

def dist(p1,p2):
    return np.linalg.norm(p1-p2)

def draw_line(img, start, vec, color):
    end = tuple(start + vec)
    cv2.line(img, tuple(start), end, color)

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle_old(v1, v2):
    cosine = dotproduct(v1, v2) / (length(v1) * length(v2)+0.0001)
    # print math.acos(cosine)/math.pi*180
    return math.acos(cosine)

def angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    a1 = math.atan2(y1, x1)
    a2 = math.atan2(y2, x2)

    # first convert angles to 0 and pi range
    a1 = math.atan2(-y1, -x1) if a1 < 0.0 else a1
    a2 = math.atan2(-y2, -x2) if a2 < 0.0 else a2

    if (a1 >= 0.5*math.pi and a2 >= 0.5*math.pi) or (a1 <= 0.5* math.pi and a2 <= 0.5*math.pi):
        v =  math.fabs(a1-a2)
    elif a1 >= 0.5*math.pi and a2 <= 0.5*math.pi:
        v = min(math.pi - a1 + a2, a1-a2)
    elif a2 >= 0.5*math.pi and a1 <= 0.5*math.pi:
        v = min(math.pi - a2 + a1, a2-a1)
    return v
    #
    # if (a1 < 0.0 and a2 < 0.0) or (a1 > 0.0 and a2 > 0.0):
    #     return math.fabs(a1-a2)
    # else:
    # return a2

def within_distance(p1, p2, distance):
    return p1 >= p2 and p1-p2 <= distance

sd = ShapeDetector()
sd.get_data('test')

# svm = sd.train_classifier()
# sd.test_classifier('test/2.jpg', svm)
