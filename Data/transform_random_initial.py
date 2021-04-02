import scipy.io as io
import os
import pickle
# 数据矩阵的组织原则为：(nSample, dim0, dim1, ...)
mat_dir = "E:\\Works\\Project\\MvCCDA\\"
file_names = ["Z.mat", "P.mat"]
file_datas = []
for name in file_names:
    file_datas.append(io.loadmat(mat_dir + name))
P = list(file_datas[1].get("P")[0])
Z = file_datas[0].get("Z")
print("strat saving data")
with open(os.getcwd() + "\\parameter_initial.pickle", "wb") as f:
    pickle.dump({"P":P, "Z":Z}, f, pickle.HIGHEST_PROTOCOL)
print("data saved")