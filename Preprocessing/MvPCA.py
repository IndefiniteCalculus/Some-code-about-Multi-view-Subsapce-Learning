import numpy as np
from . import MvProjection
# from .PCA import PCA
from sklearn.decomposition import PCA
def MvPCA(MvData, reducted_dim):
    num_views = len(MvData)
    for data in MvData:
        # a = PCA(data, reducted_dim, mode="SVD")
        # b = PCA(data, reducted_dim, mode= "standard")
        pca = PCA(n_components=reducted_dim)
        pca.fit(data)
        pass