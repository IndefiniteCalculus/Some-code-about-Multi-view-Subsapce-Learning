import numpy as np
import scipy.io as io
import matlab.engine
from Data import dataloader
# TODO: complete data transform task, the module aim to save data to mat, start matlab engine load in data and run matlab file to reduce the dim of data
# TODO: save data to mat
def save_pydata2mat(data:list, swap_dir):
    count = 0
    data_dict = {}
    mat = None
    # the data will be stack together in row by row
    for view_data in data:
        if mat is None:
            mat = view_data
        else:
            mat = np.concatenate((mat, view_data), axis=0)
    num_view = len(data)
    data_dict={"data":mat, "num_view":num_view}
    io.savemat(swap_dir + "\\temp.mat", data_dict)

if __name__ == "__main__":
    train_data, test_data, label = dataloader.load_data("pie")
    # 将数据保存为mat文件
    save_pydata2mat(train_data, "E:\\Works\\数据集")
    # 运行matlab代码对数据进行转换，覆写temp.mat
    engine = matlab.engine.start_matlab()
    engine.preprocess(nargout=0)
    # 从matlab存储的文件中读出数据矩阵
    pca_data = io.loadmat("E:\\Works\\数据集" + "\\temp.mat")
    pass