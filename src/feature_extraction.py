""" feature_extraction.py: feature extraction with PCA

    Quick tutorial on how to extract features from an 2 dimension numpy array.
    
    PCA
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as PCA
import sklearn.discriminant_analysis as FactorAnalysis

class Extractor:
    def __init__(self, data) -> None:
        """
        Args:
            data(np.array): 
                data to be extracted.
                
        Attributes:
            pca (function): 
                do pca and display the result compared to the original image
        """
        self.data = data
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
    def pca(self, p=2) -> None:
        """
        Args:
            p (int): number of principal components, default is 3.
            
        using sklearn.decomposition.PCA for feature extraction
        """
        # pca
        self.pca = PCA.PCA(n_components=p)
        self.pca_result = self.pca.fit_transform(self.data)
        
        plt.title("original data with pca components")
        plt.scatter(self.data[:, 0], self.data[:, 1], alpha=0.5, label="samples")
        for i, (comp, var) in enumerate(zip(self.pca.components_, self.pca.explained_variance_)):
            comp = comp * var  # scale component by its variance explanation power
            plt.plot(
                [self.pca.mean_[0], comp[0] + self.pca.mean_[0]],
                [self.pca.mean_[1], comp[1] + self.pca.mean_[1]],
                label=f"Component {i}",
                linewidth=5,
                color=f"C{i + 2}",
            )
        plt.show()
        
        # plot the transformed data
        plt.title("transformed data with PCA")
        plt.scatter(self.pca_result[:,0], self.pca_result[:,1])
        plt.show()
        
        
def load_data() -> np.array:
    # linear data with noise following y = x + 1
    data = np.random.rand(100,2)
    data[:,1] = data[:,0] + 1 + np.random.normal(0, 0.1, 100)
    return data    


def plot_data(data):
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    
    
def main():
    data = load_data()
    plot_data(data)
    extractor = Extractor(data)
    extractor.pca(2)

if __name__=="__main__":
    np.random.seed(24)
    main()