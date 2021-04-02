import numpy as np
import scipy.io as io
import os
import pickle
# 数据矩阵的组织原则为：(nSample, dim0, dim1, ...)
mat_dir = "E:\\Works\\Project\\MvCCDA\\"
file_names = ["reduced_te_data.mat", "reduced_tr_data.mat"]
file_datas = []
for name in file_names:
    file_datas.append(io.loadmat(mat_dir + name))

Te_data, Tr_data = file_datas[0].get("Te_data"), file_datas[1].get("Tr_data")

num_views = Tr_data.shape[-1]
Tr_views, Te_views = [],[]
for view_id in range(num_views):
    Tr_views.append(Tr_data[0][view_id].T)
    Te_views.append(Te_data[0][view_id].T)

dim_reduced_data = {"train":Tr_views, "test":Te_views}

with open(os.getcwd() + "\\dim_reduced_data.pickle", "wb") as f:
    pickle.dump(dim_reduced_data, f, pickle.HIGHEST_PROTOCOL)

