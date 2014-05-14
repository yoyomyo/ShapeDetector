__author__ = 'yoyomyo'

import pdb

try:
    import svgwrite
except ImportError:
    # if svgwrite is not 'installed' append parent dir of __file__ to sys.path
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/svgwrite'))

import svgwrite
from svgwrite import px, deg

# this class includes some logic to align the shapes

class SVGGenerator:

    CIRCLE,ELLIPSE,RECT,TRIANGLE, LINE = 0,1,2,3,4
    ALIGNMENT_DRIFT_THRESHOLD = 5
    SVG_BOUNDARY = 6
    DEBUG = True

    def __init__(self, shapes, texts):
        self.shapes = shapes
        self.texts = texts

    # given a list of shapes, draw the shape in svg
    # and save the svg to a file in the end
    def generate_svg(self, filename, width, height):
        dwg = svgwrite.Drawing(filename, size = ((width+self.SVG_BOUNDARY)*px, (height+self.SVG_BOUNDARY)*px))

        # dwg.add(dwg.rect(insert=(3*px, 3*px), size=(width*px, height*px), fill='white', stroke='black', stroke_width=3))
        for entry in self.shapes:
            shape = entry['shape']
            if shape == self.CIRCLE:
                cx, cy = entry['center']
                r = entry['radius']
                circle = dwg.add(dwg.circle(center=(cx*px, cy*px),r=r*px))
                circle.fill('none').stroke('black')
            if shape == self.ELLIPSE:
                center,size, degree = entry['ellipse']
                width, height = size
                ellipse = dwg.add(dwg.ellipse(center=(center[0]*px, center[1]*px), r=(0.5*height*px,0.5*width*px)))
                ellipse.fill('none').stroke('black')
                ellipse.rotate(degree-90, center=center)
            if shape == self.RECT:
                pts = entry['points']
                rect = dwg.add(dwg.polyline(pts, stroke = 'black', fill='none'))
            if shape == self.TRIANGLE:
                points = entry['points']
                triangle = dwg.add(dwg.polyline(points, stroke = 'black', fill='none'))
            if shape == self.LINE:
                points = entry['points']
                line = dwg.add(dwg.line(points[0],points[1],stroke = 'black'))

        for text_area in self.texts:
            paragraph = dwg.add(dwg.g(font_size= text_area.bottom - text_area.top))
            paragraph.add(dwg.text('text', ((text_area.left+self.SVG_BOUNDARY)*px, (text_area.top+self.SVG_BOUNDARY)*px)))

        dwg.save()

    def test(self, name):
        w,h = 700, 900
        dwg = svgwrite.Drawing(filename=name, size=(w, h))
        dwg.add(dwg.polyline([(201, 773), (414, 563), (178, 429),(201, 773)], stroke = 'black', fill='none'))
        dwg.save()
