import numpy as np
import scipy.io as io
import os
import pickle
# 数据矩阵的组织原则为：(nSample, dim0, dim1, ...)
mat_dir = "E:\\Works\\Project\\MvCCDA\\"
file_name = "pie"
file_data = io.loadmat(mat_dir + file_name + ".mat")

Tr_data, Tr_Labels, Te_data, Te_Labels = \
file_data.get("Tr_data"), file_data.get("Tr_Labels"), \
file_data.get("Te_data"), file_data.get("Te_Labels")

num_views = Tr_data.shape[-1]
Tr_views, Te_views = [],[]
for view_id in range(num_views):
    Tr_views.append(Tr_data[0][view_id].T)
    Te_views.append(Te_data[0][view_id].T)

Tr_Class, Te_Class = [], []
for view_id in range(num_views):
    Tr_Class.append(Tr_Labels[0][view_id].T)
    Te_Class.append(Te_Labels[0][view_id].T)

row_data = {"train":Tr_views, "test":Te_views}
labels = {"train":Tr_Class, "test":Te_Class}

with open(os.getcwd() + "\\row_data.pickle", "wb") as f:
    pickle.dump(row_data, f, pickle.HIGHEST_PROTOCOL)

with open(os.getcwd() + "\\labels.pickle", "wb") as f:
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
pass