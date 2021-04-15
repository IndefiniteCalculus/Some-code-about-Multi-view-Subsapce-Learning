import numpy as np
from Data import dataloader
import scipy.io as io
from Preprocessing import DataType_Transformation as pca_loader
import rcca
import MvCCDA_OOP
MvCCDA = MvCCDA_OOP.MvCCDA

tr_MvData, te_MvData, labels = dataloader.load_data("matlabpca")
tr_label = labels.get("train")
te_label = labels.get("test")
pca_dim = 80
# tr_MvData, va_MvData, te_MvData, tr_label, va_label, te_label = dataloader.load_data("pca_mnist-usps", pca_dim)

# the pca procedures of matlab is using mean removed data do the pca transformation, and the fit of W from each view
# is used to construct an only W_mean do the data dim reduction
# TODO: try implement multiview pca on python when has free time
from Preprocessing import MvPCA

# train_data = MvPCA.MvPCA(train_data, pca_dim)
# test_data = MvPCA.MvPCA(test_data, pca_dim)
# initialization of parameters to be learned, size of row data should be (num_sample, num_dim)

# mnist_usps tr te split
# mnist_usps_mat = io.loadmat("E:\\Works\\数据集\\mnist-usps\\mnist_usps.mat")
# tr_MvData, tr_label = mnist_usps_mat.get('Tr_data'), mnist_usps_mat.get('Tr_Labels')
# te_MvData, te_label = mnist_usps_mat.get('Te_data'), mnist_usps_mat.get('Te_Labels')
# tr_MvData, tr_label = pca_loader.cell2list(tr_MvData), pca_loader.cell2list(tr_label)
# te_MvData, te_label = pca_loader.cell2list(te_MvData), pca_loader.cell2list(te_label)
# tr_MvData,te_MvData  = pca_loader.load_pca(tr_MvData, te_MvData, pca_dim)
print("data load in, start training")
param = io.loadmat("E:\\Works\\MatlabWork\\MvCCDA_mnist_usps\\param.mat")
common_comp = param.get('Z').T
map_matrices = param.get('P')
map_matrices = pca_loader.cell2list(map_matrices)
# param = dataloader.load_param()
# common_comp = param.get("Z").T
# map_matrices = param.get("P")

# mnist-usps tr 70 te 30
# algorithm="LPDP", t = 1, sigma = 200, lambda1=0.06, lambda2=0.02, lambda3=5e-4, lambda4=5e-4 #  7-10
# algorithm="LPP", sigma = 200, lambda1=0.06, lambda2=0.02, lambda3=5e-3 82.33 # 19
# algorithm="LPP", t = 1, sigma = 200, lambda1=0.00, lambda2=0.00, lambda3=0, lambda4=0 33.83 #
#
# mnist-usps tr 50 te 25
# algorithm="LPDP", t = 1, sigma = 200, lambda1=0.06, lambda2=0.2, lambda3=5e-5, lambda4=5e-5 # 6s 81.8%
# algorithm="LPP", sigma = 200, lambda1=0.06, lambda2=0.2, lambda3=5e-3 # 5s 81.6%
# algorithm="LPDP", t = 1, sigma = 200, lambda1=0.6, lambda2=0.2, lambda3=0, lambda4=5e-4 #7-8s 82.2%
#
# pie                                                                           # time acc nmi
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-5 # 19s 72.3% 89.6%
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-4 # 19s 72.3% 89.6%
#
# algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-5, lambda4=5e-5 # 20s 72.44% 89.6%
# algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-5 # 20s 72.44% 89.6%
#
# algorithm="MMC", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 60s 77.99% 91.4%
# algorithm="MMMC", t = 0.1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 41s 75.33% 90.5%
# algorithm="MMMC", t = 0.01, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 40s 74.67% 90.4%
# algorithm="MMMC", t = 1.5, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 75s 78.75% 91.5%
# algorithm="MMMC", t = 2, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 87s 79.34% 91.8%
#
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0 # 20s 72.34% 89.6%
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.06, lambda2=0.02, lambda3=0 # 22s 49.56% 80.93%
#
import time

model = MvCCDA(
    algorithm="LPP", t=1, sigma=2000, lambda1=0.6, lambda2=0.02, lambda3=0  # 22s 49.56%
)

t1 = time.time()
map_matrices, common_comp = model.train(tr_MvData, tr_label, rand_seed=None)
t2 = time.time()
t = t2 - t1
print("training done")

# project test data to subspace
ave_acc, ave_nmi = model.test(te_MvData, te_label, map_matrices, mode="pair_wise")
print("time: " + str(t))
print("ave_acc: " + str(ave_acc) + " ave_nmi: " + str(ave_nmi))
#
# print("percentage of how many vector and it's nearest vector shared with same label among unmapped dataset: \n"+str(unmapped_ave_acc)+"%")
# print("percentage of how many vector and it's nearest vector shared with same label among mapped dataset: \n" + str(mapped_ave_acc)+"%")
pass
