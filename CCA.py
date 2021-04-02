import numpy as np
from Data import dataloader
from scipy.linalg import eigh
from Preprocessing.MvPCA import MvPCA
def get_crosscovs(train_data):
    # warning: 可能把数据向量的维度和数据向量的个数弄反了
    # 获取各个视角上数据之间的协方差矩阵，用列表的方式存储，
    # 列表的索引顺序与视角i和j之间协方差矩阵的关系为：index = i*row+j（从0开始计数）
    # 矩阵形状为：视角维度x视角维度
    crosscovs = []
    for i in train_data:
        for j in train_data:
            crosscovs.append(np.dot(i.T, j))
    return crosscovs

def generaleig_matrix(crosscovs, num_views, num_dims):
    # 生成广义特征值问题的矩阵
    cov_shape = (sum(num_dims),sum(num_dims))
    R_matrix, L_matrix = np.zeros(cov_shape), np.zeros(cov_shape) # 右矩阵对角线上为视角间自协方差矩阵，左矩阵对角线外为交叉协方差矩阵
    # 求解广义特征值问题要求左右矩阵正定，所以如果矩阵不满秩，那么就不是正定，问题演变为ill-pose问题，需要加上对角线元素保证矩阵正定才能继续求解

    for i in range(num_views):
        xyrange = (sum(num_dims[:i]),
                   sum(num_dims[:i]) + num_dims[i])
        R_matrix[xyrange[0]: xyrange[1], xyrange[0]: xyrange[1]] = crosscovs[i*num_views + i]
        for j in range(num_views):
            if i != j:
                xrange = (sum(num_dims[:i]),
                          sum(num_dims[:i]) + num_dims[i])
                yrange = (sum(num_dims[:j]),
                          sum(num_dims[:j]) + num_dims[j])
                L_matrix[xrange[0]:xrange[1], yrange[0]:yrange[1]] = crosscovs[i*num_views + j]

    L_matrix = (L_matrix.T + L_matrix) / 2
    R_matrix = (R_matrix.T + R_matrix) / 2
    R_matrix = R_matrix + regularization * np.eye(R_matrix.shape[0], R_matrix.shape[1])
    # a = np.linalg.det(R_matrix)
    return L_matrix, R_matrix

subspace_dim = 90
regularization = 0 # 7.535 # 使用regularization对角化 https://www.sciencedirect.com/science/article/abs/pii/0304407676900105?via%3Dihub
if __name__ == "__main__":
    train_set, test_set, _ = dataloader.load_data("matlabpca")
    train_data = train_set
    # 获取PCA降维后的数据向量
    # MvPCA(train_data,80)

    # 构建协方差矩阵
    crosscovs = get_crosscovs(train_data)

    # 从数据中获取参数
    num_view = len(train_data)
    num_dims = [view.shape[-1] for view in train_data]
    max_subspace_dim = sum(num_dims)

    # 构建广义特征值问题的求解矩阵并求解
    L_matrix, R_matrix = generaleig_matrix(crosscovs, num_view, num_dims)
    eigvals, eigvectors = eigh(L_matrix, R_matrix, eigvals = (max_subspace_dim - subspace_dim, max_subspace_dim - 1))

    # 对特征向量进行排序（因为特征值大小反映正相关度大小，CCA的目标就是要让两个数据经过某种向量变换后的相关度最大，这里的变换通过特征向量实现）
    eig_idx = np.argsort(eigvals)[::-1]
    ordered_eigvectors = eigvectors[:,eig_idx]

    # 取出将每个视角数据映射到目标子空间的映射向量
    views_mapping_vectors = []
    for i in range(num_view):
        eigvector_range = (sum(num_dims[:i]), sum(num_dims[:i]) + num_dims[i])

        views_mapping_vectors.append(
            ordered_eigvectors[eigvector_range[0]:eigvector_range[1],:]
        )

    pass