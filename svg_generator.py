__author__ = 'yoyomyo'

import pdb
import svgwrite
from svgwrite import px, deg

# this class includes some logic to align the shapes

class SVGGenerator:

    CIRCLE,ELLIPSE,RECT,TRIANGLE = 0,1,2,3
    ALIGNMENT_DRIFT_THRESHOLD = 5
    SVG_BOUNDARY = 6
    DEBUG = True

    def __init__(self, shapes):
        self.shapes = shapes

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
            elif shape == self.TRIANGLE:
                points = entry['points']
                triangle = dwg.add(dwg.polyline(points, stroke = 'black', fill='none'))

        dwg.save()

    def test(self, name):
        w,h = 700, 900
        dwg = svgwrite.Drawing(filename=name, size=(w, h))
        dwg.add(dwg.polyline([(201, 773), (414, 563), (178, 429),(201, 773)], stroke = 'black', fill='none'))
        dwg.save()
