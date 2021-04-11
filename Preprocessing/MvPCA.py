import numpy as np
# from Preprocessing.PCA import PCA
from sklearn.decomposition import PCA
# from sklearn.decomposition import KernelPCA
def MvPCA(MvData, reducted_dim):
    num_views = len(MvData)
    reduced_MvData = []
    for data in MvData:
        pca = PCA(n_components=reducted_dim,svd_solver="full")
        reduced_MvData.append(pca.fit_transform(data))
    return reduced_MvData