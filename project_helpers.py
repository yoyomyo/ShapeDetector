__author__ = 'yoyomyo'

import math
import numpy as np
import cv2

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

NESTED_CONTOUR_DISTANCE = 15    #if a two contours are nested and their boundaries
                                # are within 10px then remove the innner conotur
PT_DISTANCE_THRESHOLD = 20      # this distance may change, threshold should depend on how big the contour is

ALIGNMENT_DRIFT_THRESHOLD = 5


def filter_contours(contours):
    contours = filter(lambda cnt: len(cnt) > CONTOUR_THRESHOLD, contours)
    contours = remove_nested_contours(contours)
    return contours

def connect_contours(contours):
    contours = filter(lambda cnt: len(cnt) > CONTOUR_THRESHOLD, contours)
    contours = merge_contours(contours)
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
                flag =  are_nested_contour(contour_table[k], to_add)
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


def merge_contours(contours):
    merged = []
    boundingBoxes = []
    for cnt in contours:
        x1,y1,w1,h1 = cv2.boundingRect(cnt)
        for box in boundingBoxes:
            x2,y2,w2,h2 = box

    return merged.values()

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

def show_image_in_window(win_name, img):
    cv2.imshow(win_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()