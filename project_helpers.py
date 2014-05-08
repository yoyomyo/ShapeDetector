__author__ = 'yoyomyo'

import math
import numpy as np
import cv2
from operator import itemgetter
from bisect import *
import pdb
# global constants

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

NESTED_CONTOUR_DISTANCE = 15    # if a two contours are nested and their boundaries
                                # are within 10px then remove the innner conotur
PT_DISTANCE_THRESHOLD = 20      # this distance may change, threshold should depend on how big the contour is

ALIGNMENT_DRIFT_THRESHOLD = 0.1

TABLE_BOX_DISTANCE = 10






# contour related

def filter_contours(contours, w, h):
    # filter the contours based on the dimension of the img
    contours = filter(lambda cnt: len(cnt) > 0.05*min(w,h), contours)
    contours = remove_nested_contours(contours)
    return contours

def remove_nested_contours(contours):
    contour_table = {}
    for i, cnt in enumerate(contours):
        to_add = get_contour_extreme_points(cnt)
        if len(contour_table) == 0:
            contour_table[i] = to_add
        else:
            to_replace = i
            for k in contour_table:
                flag = are_nested_contour(contour_table[k], to_add)
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
def are_nested_contour(extreme_points1, extreme_points2):
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

    a = within_contour_distance(xl1, xl2)
    b = within_contour_distance(xr2, xr1)
    c = within_contour_distance(yt1, yt2)
    d = within_contour_distance(yb2, yb1)

    e = within_contour_distance(xl2, xl1)
    f = within_contour_distance(xr1, xr2)
    g = within_contour_distance(yt2, yt1)
    h = within_contour_distance(yb1, yb2)

    if e and f and g and h:
        return 1
    elif a and b and c and d:
        return 2
    return 3

def within_contour_distance(p1, p2):
    return within_distance(p1,p2,NESTED_CONTOUR_DISTANCE)

def within_distance(p1, p2, distance):
    return p1 >= p2 and p1-p2 <= distance

def get_contour_extreme_points(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    return [leftmost, rightmost, topmost, bottommost]

def get_contour_centroid(cnt):
    # compute the centroid of the contour to help compute chord normals and orientation
    M = cv2.moments(cnt)
    centroid_x = int(M['m10']/M['m00'])
    centroid_y = int(M['m01']/M['m00'])
    return np.array([centroid_x,centroid_y])

# used to get text regions in a sketch
def connect_contours(contours):
    contours = filter(lambda cnt: len(cnt) > CONTOUR_THRESHOLD, contours)
    contours = merge_contours(contours)
    return contours

def merge_contours(contours):
    merged = []
    boundingBoxes = []
    for cnt in contours:
        x1,y1,w1,h1 = cv2.boundingRect(cnt)
        for box in boundingBoxes:
            x2,y2,w2,h2 = box
    return merged.values()


# return true if rect1 is within rect2
# or rect2 is within rect1
def are_nexted_rectangles(rect1, rect2):
    left1, right1, top1, bottom1 = rect1
    left2, right2, top2, bottom2 = rect2
    one = left1<left2 and right1 > right2 and top1 < top2 and bottom1 > bottom2
    two = left2<left1 and right2 > right1 and top2 < top1 and bottom2 > bottom1
    return one or two

# return coordinates of the box if box is horizontal
# else return empty array
def is_horizontal_box(box):
    a,b,c,d = sorted(box, key=itemgetter(0,1))
    e = abs(a[0] - b[0]) < ALIGNMENT_DRIFT_THRESHOLD * a[0]
    f = abs(c[0] - d[0]) < ALIGNMENT_DRIFT_THRESHOLD * c[0]
    g = abs(a[1] - c[1]) < ALIGNMENT_DRIFT_THRESHOLD * a[1]
    h = abs(b[1] - d[1]) < ALIGNMENT_DRIFT_THRESHOLD * b[1]
    if e and f and g and h:
        left = (a[0] + b[0])/2
        right = (c[0] + d[0])/2
        top = min ( (a[1] + c[1])/2, (b[1] + d[1])/2)
        bottom = max ( (a[1] + c[1])/2, (b[1] + d[1])/2)
        return [ [left,top], [right,top], [right, bottom], [left,bottom] ]

    else:
        return []

def overlap_boxes(box1, box2):
    l1 = box1[0][0]
    r1 = box1[2][0]
    t1 = box1[0][1]
    b1 = box1[2][1]

    l2 = box2[0][0]
    r2 = box2[2][0]
    t2 = box2[0][1]
    b2 = box2[2][1]

    a = abs(l1-r2) < TABLE_BOX_DISTANCE and abs(t1-t2) < TABLE_BOX_DISTANCE
    b = abs(r1-l2) < TABLE_BOX_DISTANCE and abs(t1-t2) < TABLE_BOX_DISTANCE
    c = abs(b1-t2) < TABLE_BOX_DISTANCE and abs(l1-l2) < TABLE_BOX_DISTANCE
    d = abs(t1-b2) < TABLE_BOX_DISTANCE and abs(l1-l2) < TABLE_BOX_DISTANCE

    return a or b or c or d


# global helper functions
def midpoint(p1, p2):
    return (p1+p2)/2

def normalize(v):
    return v/np.linalg.norm(v)

def dist(p1,p2):
    return np.linalg.norm(p1-p2)

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

def preprocess_image(img, max_img_dim = MAX_IMG_DIM, morph_dim = MORPH_DIM):
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

def get_extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent


def draw_contours(contours, img):
    for cnt in contours:
        #cv2.drawContours(color_img,[cnt],-1,(0,255,0),1)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        #cv2.drawContours(img,[approx],-1,(0,255,0),1)

    show_image_in_window('contour', img)

# append file2 to file1
def append_result_to_file(file1, file2):
    with open(file1, "a") as f1:
        with open(file2, "r") as f2:
            f1.write(f2.read())

def show_image_in_window(win_name, img):
    cv2.imshow(win_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()