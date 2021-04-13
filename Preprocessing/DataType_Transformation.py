import numpy as np
import scipy.io as io
import matlab.engine
from Data import dataloader
# TODO: complete data transform task, the module aim to save data to mat, start matlab engine load in data and run matlab file to reduce the dim of data
# TODO: save data to mat
def save_pydata2mat(data:list, swap_dir, pca_dim):
    count = 0
    data_dict = {}
    mat = None
    view_dims = []
    # the data from each view will be stack together as (n_sample, view1_dim + view2_dim + ...)
    for view_data in data:
        if mat is None:
            mat = view_data
        else:
            mat = np.concatenate((mat, view_data), axis=1)
        view_dims.append(view_data.shape[1])

    num_view = len(data)
    data_dict={"data":mat, "num_view":num_view, "dim_list":np.array(view_dims), "pca_dim":pca_dim}
    io.savemat(swap_dir + "\\temp.mat", data_dict)

def cell2list(MvMat):
    MvData = []
    for v in range(MvMat.shape[1]):
        MvData.append(MvMat[0, v].T)
    return MvData

def load_pca(train_data, pca_dim):
    # 将数据保存为mat文件
    save_pydata2mat(train_data, "E:\\Works\\数据集", pca_dim)
    # 运行matlab代码对数据进行转换，覆写temp.mat
    engine = matlab.engine.start_matlab()
    engine.cd('Preprocessing',nargout=0)
    engine.preprocess(nargout=0)
    # 从matlab存储的文件中读出数据矩阵
    pca_data = io.loadmat("E:\\Works\\数据集" + "\\temp.mat")
    pca_data = pca_data.get('MvData')
    num_view = pca_data.shape[1]
    Mv_PCA = []
    for view_idx in range(num_view):
        Mv_PCA.append(pca_data[0, view_idx].T)
    return Mv_PCA

if __name__ == "__main__":
    dataset_name = "pie"
    train_data, test_data, label = dataloader.load_data(dataset_name)
    # 将数据保存为mat文件
    save_pydata2mat(train_data, "E:\\Works\\数据集")
    # 运行matlab代码对数据进行转换，覆写temp.mat
    engine = matlab.engine.start_matlab()
    engine.preprocess(nargout=0)
    # 从matlab存储的文件中读出数据矩阵
    pca_data = io.loadmat("E:\\Works\\数据集" + "\\temp.mat")
    pca_data = pca_data.get('MvData')
    num_view = pca_data.shape[1]
    Mv_pca = []
    for view_idx in range(num_view):
        Mv_pca.append(pca_data[0,view_idx])
    pass