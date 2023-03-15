""" gaussian_blur.py: gaussian blur with trackbars
    
    __author__: sabin lee
    __license__: MIT
    __version__: 0.1
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class blurrr:
    def __init__(self, path="./data/lena.png", style="gaussian") -> None:
        """
        Args:
            path (str, optional): 
                path to the image. 
                Defaults to "img/cat.jpeg".
            style (str, optional):
                style of the blur, can be "gaussian", "median", "bilateral".
                
        Attributes:
            process (function): 
                do gaussian blur and display the result compared to the original image 
        """
        original_image = cv.imread(path)
        self.img = original_image
        # self.img = cv.cvtColor(original_image,cv.COLOR_BGR2RGB)
        self.vectorized = self.img.reshape((-1,3))
        self.vectorized = np.float32(self.vectorized)
        self.style = style
    
    def process(self) -> None:
        """_summary_: do blur following style with trackbars
        """
        def nothing(x):
            pass

        cv.namedWindow('image')
        cv.createTrackbar('ksize','image',1,30,nothing)
        cv.createTrackbar('sigmaX','image',0,30,nothing)
        cv.createTrackbar('sigmaY','image',0,30,nothing)

        while(1):
            ksize = cv.getTrackbarPos('ksize','image')
            sigmaX = cv.getTrackbarPos('sigmaX','image')
            sigmaY = cv.getTrackbarPos('sigmaY','image')
            if ksize % 2 == 0:
                ksize += 1
            if ksize == 1:
                ksize = 3
            if sigmaX == 0:
                sigmaX = 1
            if sigmaY == 0:
                sigmaY = 1
                
            if self.style == "gaussian":
                blur = cv.GaussianBlur(self.img,(ksize,ksize),sigmaX,sigmaY)
            elif self.style == "median":
                blur = cv.medianBlur(self.img,ksize)
            elif self.style == "bilateral":
                blur = cv.bilateralFilter(self.img,ksize,sigmaX,sigmaY)
                
            cv.imshow('image',blur)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break

        cv.destroyAllWindows()
        
if __name__ == "__main__":
    blurr = blurrr(style="gaussian")
    blurr.process()