""" rgb_control.py: interactive control of RGB channels.
    
    Quick tutorial on how to use kmeans with openCV.
    adjust the RGB channels of an image by using trackbars.
    
    __author__: sabin lee
    __license__: MIT
    __version__: 0.1
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

# Create a black image, a window
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF   1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    img = np.zeros((300,512,3), np.uint8)
    
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break  
        
        
cv.destroyAllWindows()

# Path: src/rgb_control.py