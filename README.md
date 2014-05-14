ShapeDetector
=============

A program that detects circles, ellipses, triangles and rectangles in an image, and then converts them into an svg.

Dependencies:
cv2 (OpenCV Python package)

svgwrite

To run:

`python img2svg.py dir_of_input_images`

The code comes with some input images in the example_input directory:
`python img2svg.py example_input`

To turn on intermediate results: 
`python img2svg.py example_input 1`

To turn off intermediate results:
`python img2svg.py example_input 0`
