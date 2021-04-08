import numpy as np
# from Preprocessing.PCA import PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
# from sklearn.decomposition import
def MvPCA(MvData, reducted_dim):
    num_views = len(MvData)
    reduced_MvData = []
    for data in MvData:
        # map_matrix = PCA(data, reducted_dim)
        # reduced_MvData.append()

        pca = PCA(n_components=reducted_dim,svd_solver='arpack')
        reduced_MvData.append(pca.fit_transform(data))
    return reduced_MvData