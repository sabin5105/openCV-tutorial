""" image_segmentation.py: image segmentation using k-means clustering.

    Quick tutorial on how to use kmeans with openCV.
    segment an image into K clusters.
    
    __author__: sabin lee
    __license__: MIT
    __version__: 0.1
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class k_means:
    def __init__(self, path="./data/cat.jpeg") -> None:
        """
        Args:
            path (str, optional): 
                path to the image. 
                Defaults to "img/cat.jpeg".
            
        Attributes:
            process (function): 
                do k-means clustering and display the result compared to the original image 
        """
        original_image = cv.imread(path)
        self.img = cv.cvtColor(original_image,cv.COLOR_BGR2RGB)
        self.vectorized = self.img.reshape((-1,3))
        self.vectorized = np.float32(self.vectorized)

    
    def process(self, K=3) -> None:
        """
        Args:
            K (int): number of clusters, default is 3.
        """
        # it takes a while to run this code
        criteria = (cv.TERM_CRITERIA_EPS + 
			        cv.TERM_CRITERIA_MAX_ITER,10,1.0)

        attempts = 10 
        ret,label,center = cv.kmeans(self.vectorized,K,None,criteria,attempts,
                                        cv.KMEANS_PP_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        self.result_image = res.reshape((self.img.shape))
              
        figure_size = 15
        plt.figure(figsize=(figure_size,figure_size))
        plt.subplot(1,2,1),plt.imshow(self.img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(self.result_image)
        plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
        plt.show()
        
if __name__ == "__main__":
    k_means = k_means()
    k_means.process(2)
    k_means.process(5)
    k_means.process(10)