__author__ = 'yoyomyo'
import os,sys,math,pdb
import numpy as np
import cv2
from svg_generator import *
from project_helpers import *
from hough_array import *

class ShapeDetector:

    CIRCLE,ELLIPSE,RECT,TRIANGLE,LINE = 0,1,2,3,4


    MAX_IMG_DIM = 1000              # if width or height greater than 1000 pixels, processing image would be slow
    MORPH_DIM = (3,3)
    CONTOUR_THRESHOLD = 20          # if a contour does not contain more than 50 pixels,
                                    # then we are not interested in the contour
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    YELLOW = (0,255,255)
    BG = (255,255,0)

    EPS = 0.0001                    # used to prevent division by zero

    NESTED_CONTOUR_DISTANCE = 15    # if a two contours are nested and their boundaries
                                    # are within 15px then remove the innner conotur

    PT_DISTANCE_THRESHOLD = 20      # point distance threshold, used in get_key_points


    # # this function is not called
    # def get_data(self, dir):
    #     i = 0
    #     for root, subdirs, files in os.walk(dir):
    #         for file in files:
    #             if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
    #                 path_to_img = os.path.join(root,file)
    #                 print path_to_img
    #                 img = cv2.imread(path_to_img)
    #                 shapes = self.get_shapes_from_img(img, i)
    #                 h,w,c = img.shape
    #                 i += 1
    #                 self.generate_svg(shapes, 'test'+ str(i) +'.svg', w, h)
    #
    # # get shape features from img, not called
    # def get_shapes_from_img(self,  contours, img, num):
    #     color, bw_img = preprocess_image(img)
    #     shapes = self.get_shapes(contours, color, bw_img)
    #     return shapes

    def get_shapes(self, contours, color_img, bw_img, text_regions):
        #contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        w,h = bw_img.shape
        contours = filter_contours(contours, w, h)
        #draw_contours(contours, color_img)
        mask1 = np.zeros(bw_img.shape,np.uint8) #interest points
        mask2 = np.zeros(bw_img.shape,np.uint8) #inflection points

        shapes = [] # a shape is a dictionary containing necessary information to draw the shape in SVG
        rectangles = [] # this will be used to detect tables
        all_points = []
        for h, cnt in enumerate(contours):
            # if cnt fall into one of the text_regions, disregard this cnt
            points = self.get_key_points(cnt, bw_img, mask1, mask2)

            count = 0
            for pt in points:
                for region in text_regions:
                    if region.contains(pt[0]):
                        count += 1

            if count > 0.5 * len(points):
                continue # the points are in the text region
            else:
                shape = self.infer_shape_from_key_points(points, cnt)

                if shape == self.CIRCLE:
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(color_img,center,radius,self.RED,2)
                    shapes.append({
                        'shape':self.CIRCLE,
                        'center':(int(x),int(y)),
                        'radius':int(radius),
                        'points':points
                    })
                elif shape == self.ELLIPSE:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(color_img,ellipse,self.GREEN,2)
                    shapes.append({
                        'shape':self.ELLIPSE,
                        'ellipse': ellipse,
                        'points':points
                    })

                elif shape == self.RECT:
                    x,y,w,h = cv2.boundingRect(cnt)
                    # cv2.rectangle(color_img, (x,y), (x+w,y+h), self.YELLOW, 2)
                    all_points += points
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.cv.BoxPoints(rect)

                    # make the box horizontal if the box is almost horizontal
                    hor_box = is_horizontal_box(box)
                    box = hor_box if hor_box else box
                    box = np.int0(  (np.array(box)) )

                    # keep a global copy of all rectangles for later filtering
                    rectangles.append( (box, (x,x+w,y,y+h)) )

                    # shapes.append({
                    #     'shape':self.RECT,
                    #     'points':[tuple(pt) for pt in box]+[tuple(box[0])],
                    # })


                elif shape == self.TRIANGLE:
                    # pdb.set_trace()
                    points = np.int0(points)
                    try:
                        cv2.drawContours(color_img,[points],0,self.BLUE ,2)
                    except cv2.error:
                        print "unable to draw triangle"
                    shapes.append({
                        'shape':self.TRIANGLE,
                        'points':[tuple(pt[0]) for pt in points] + [tuple(points[0][0])]
                    })
        # show_image_in_window('interest points', mask1)
        # show_image_in_window('inflection points', mask2)

        all_points = np.int0(all_points)
        # for pt in all_points:
        #     cv2.circle(color_img, tuple(pt[0]),3, self.RED, thickness=2, lineType=8, shift=0)
        #     #txt = "%.2f" % (d/math.pi*180)
        #     #cv2.putText(color_img, txt, tuple(current[0]+[4,4]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, self.RED, 1, 8)

        #
        # for region in text_regions:
        #     if region.is_text_region():
        #         cv2.rectangle(color_img,(region.left,region.top),(region.right,region.bottom),GREEN,1)



        # the problem now is that the big table rectangle
        if self.find_table_exist(rectangles, bw_img):
            # if there is a table, use find_table to build a table from all points
            box, lines = self.find_table(all_points, w, h)

            for line in lines:
                shapes.append({
                    'shape':self.LINE,
                    'points': line
                })
                cv2.line(color_img, line[0], line[1], self.BG ,2)

            # filter out the shapes that overlaps with the table
            include_shapes = []
            for shape in shapes:
                points = shape['points']
                if not shape['shape'] == self.LINE and self.contains(box, points):
                    pass
                else:
                    include_shapes.append(shape)
            shapes = include_shapes

            # then add the non-table rectangles to the image
            for rect, bound in rectangles:
                if not self.contains(box, rect) \
                        and not self.contains(bound, \
                                              [(box[0],box[2]),(box[1],box[2]),(box[0],box[3]), (box[1],box[3])]):
                    shapes.append({
                        'shape':self.RECT,
                        'points':[tuple(pt) for pt in rect]+[tuple(rect[0])],
                    })
                    try:
                        cv2.drawContours(color_img,[rect],0,self.YELLOW,2)
                    except cv2.error:
                        print "unable to draw rectangle"
        else:
            for rect, bound in rectangles:
                shapes.append({
                    'shape':self.RECT,
                    'points':[tuple(pt) for pt in rect]+[tuple(rect[0])],
                })
                try:
                    cv2.drawContours(color_img,[rect],0,self.YELLOW,2)
                except cv2.error:
                    print "unable to draw rectangle"
        return shapes, color_img

    # return true if a box contains a shape
    def contains(self, box, points):
        left, right, top, bottom = box
        count = 0
        for pt in points:
            if len(pt) == 2:
                x,y = pt
            else:
                x,y = pt[0]
            if x >= left and x <= right and y >= top and y <= bottom:
                count += 1
        return count >= 1


    def get_key_points(self, cnt, bw_img, black_img1, black_img2):
        previous_gradient = None
        diffs = []
        mask = np.zeros(bw_img.shape,np.uint8)
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
                # if diff is greater than 300 degree, the gradient change is too big, set it to zero
                diff = 0 if diff > 300.0/180.0*math.pi else diff

                c = diff*255 #intensity
                diffs.append(diff)
                cv2.line(black_img1, tuple(prev[0]), tuple(nxt[0]), (c,c,c), thickness=1, lineType=8, shift=0)
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

                cv2.circle(black_img1, tuple(cnt[j+1][0]),3, (255,255,255), thickness=2, lineType=8, shift=0)


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
                   cv2.circle(black_img2, tuple(current[0]),3, (255,255,255), thickness=2, lineType=8, shift=0)
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

    #a primitive way for finding boxes
    def find_table_exist(self, rectangles, img):
        h,w = img.shape
        table = {}

        # this is to help find the locations of horizontal and vertical lines
        hor = []
        ver = []
        for i,_ in enumerate(rectangles):
            table[i] = []

        for i,val1 in enumerate(rectangles):
            box1, _ = val1
            for j,val2 in enumerate(rectangles):
                box2, _ = val2
                if j<=i: continue
                if overlap_boxes(box1, box2):
                    table[i].append(box2)
                    table[j].append(box1)

        for k in table:
            box = rectangles[k][0]
            if len(table[k])>=2:
                for pt in box:
                    ver.append(pt[0])
                    hor.append(pt[1])

        hor = np.array(hor)
        ver = np.array(ver)
        hor_bin = [i for i in range(0,h,30)]
        ver_bin = [i for i in range(0,w,30)]
        hor_hist, _ = np.histogram(hor, bins=hor_bin)
        ver_hist, _ = np.histogram(ver, bins=ver_bin)

        hor = [ (hor_bin[i] + hor_bin[i+1])/2 for i,x in enumerate(hor_hist) if x > 2 ]
        ver = [ (ver_bin[i] + ver_bin[i+1])/2 for i,x in enumerate(ver_hist) if x > 2 ]

        return hor and ver


    def find_table(self, points, w, h):

        # points: all key points for the table
        # each point vote for a horizontal line and a verticle line
        # in the end, need to find a way to collect the majority votes
        # if no table is found, return empty line

        x_data = HoughArray(30)    # vertical lines, with an ambiguity of 30px
        y_data = HoughArray(30)    # horizontal lines, with an ambiguity of 30px

        for pt in points:
            x,y = pt[0]
            x_data.add(x)
            y_data.add(y)

        lines = []
        # pick out the lines
        ver_line_pos = x_data.get_high_votes()
        hor_line_pos = y_data.get_high_votes()

        if not ver_line_pos or not hor_line_pos:
            return lines

        # now need to create the actual lines from the positions
        left =  min(ver_line_pos)
        right = max(ver_line_pos)
        top = min(hor_line_pos)
        bottom = max(hor_line_pos)

        for x in ver_line_pos:
            lines.append( [(x,top), (x,bottom)] )
        for y in hor_line_pos:
            lines.append( [(left,y), (right,y)] )

        box = [left, right, top, bottom]
        return box, lines


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
