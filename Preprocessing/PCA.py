import numpy as np
import time
def PCA(data, reducted_dim, mode="standard"):
    if mode == "standard":
        # get coordinate formed by eigen vectors which ordered by eigen values
        t1 = time.time()
        cov = np.dot(data.T, data)
        cov = cov / (data.shape[0] - 1) # unbiased estimation
        t2 = time.time()
        eig_val, eig_vec = np.linalg.eig(cov)
        t3 = time.time()
        eig_idx = np.argsort(abs(eig_val))[::-1]
        eig_val = eig_val[eig_idx]
        eig_vec = eig_vec[:,eig_idx]
        return eig_val, eig_vec
        # projection
        # the data should be organized in form like: (nSamples, dim)
        coordinates = eig_vec[:,:reducted_dim]
    elif mode == "SVD":
        # svd decomposition
        U, diag_elements, V = np.linalg.svd(data / np.sqrt(data.shape[0] - 1))
        # diag_matrix = np.power(diag_elements, 2) * np.eye(diag_elements.shape[0], diag_elements.shape[0])
        eig_idx = np.argsort(abs(diag_elements))[::-1]
        eig_val = diag_elements[eig_idx]
        eig_vec = U[:,eig_idx]
        return eig_val, eig_vec
        pass

    time1 = t2 - t1
    time2 = t3 - t1
    pass