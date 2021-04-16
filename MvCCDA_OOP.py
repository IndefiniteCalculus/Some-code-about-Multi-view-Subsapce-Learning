import numpy as np
from Data import dataloader
import scipy.io as io
from Preprocessing import DataType_Transformation as pca_loader
import rcca

class MvCCDA():
    def __init__(self, alpha=2.3849, lambda1=0.6, lambda2=0.002, lambda3=0.0005,
                 sigma=2000, subspace_dim=None, algorithm="LPP", lambda4=None, t=1):
        # initial parameters of hyper setting
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.sigma = sigma
        self.subspace_dim = subspace_dim
        self.lambda4 = lambda4
        self.t = t  # the coefficient of Sb-tSw
        self.algorithm = algorithm
        if self.algorithm == "MMC" or self.algorithm == "MMMC":
            self.lambda3 = 0
            if self.algorithm == 'MMC':
                self.t = 1
            self.algorithm = 'LPDP'
        pass


    def train(self, train_data, train_labels,
              common_comp: np.array = None, map_matrices: list = None, rand_seed=None,
              training_mode:str = "normal", valid_rate = 1, using_view = "all",
              z_loss = 1e-4, z_count = 200, p_loss = 1e-4, p_count = 200, total_loss = 5e-7, total_count = 200):

        # initial parameters about input data
        self.num_sample = self._obtain_numsample(train_data[0])
        self.num_dim = self._obtain_numdim(train_data[0])
        self.num_view = self._obtain_numview(train_data)
        self.num_class, self.max_class, self.min_class = self._obtain_numclass(train_labels)
        if self.subspace_dim is None:
            self.subspace_dim = self.num_class
        # initial matrices which will be used in the update procedures
        self.onehots = self._num2onehot(train_labels)

        self.hotkernel_mat = self._get_hotkernel_mat(train_labels, train_data)

        eks = self.onehots[0].T
        adj_mat_sum = np.zeros((self.num_sample, self.num_sample))
        for c_idx in range(self.num_class):
            ek = eks[c_idx, :]
            adj_mat_k = np.outer(ek, ek)
            adj_mat_sum = adj_mat_sum + adj_mat_k
        I = np.eye(self.num_sample)
        e = np.ones(self.num_sample)
        self.M = (I - np.outer(e, e)) / self.num_sample - (1 + self.t) * (I - adj_mat_sum)

        # initial common comp and map matrices
        if rand_seed is not None:
            np.random.seed(rand_seed)

        if common_comp is None:
            common_comp = np.random.random((self.num_sample, self.subspace_dim))
        else:
            pass

        if map_matrices is not None:
            if len(map_matrices) != self.num_view:
                exit("the num of map matrices " + str(len(map_matrices)) +
                     " is not equal to num_view " + str(self.num_view) + " obtained from dataset")
            for map in map_matrices:
                if map.shape != (self.subspace_dim, self.num_dim):
                    exit("one of shape of mapping matrix among the map matrices does"
                         "not satisfy the mapping acquirement of dataset")
        else:
            map_matrices = []
            for i in range(self.num_view):
                map_matrices.append(np.random.random((self.subspace_dim, self.num_dim)))

        # normally update common component and map matrices
        if training_mode == "normal":
            return self._optimization(train_data, common_comp, map_matrices,
                                      z_loss, z_count, p_loss, p_count, total_loss, total_count)
        if training_mode == "cross_validation":
            # split samples into 10 separated parts
            each_n_sample = self.num_sample // 10
            unfull_num = self.num_sample % 10

            # generate indices of samples
            slice_indices = []
            for i in range(10):
                slice_indices.append([i * each_n_sample, (i+1) * each_n_sample])

            # balance the proportion of split sample sets
            if unfull_num > 5:
                used_unfull_num = 0
                for indices in slice_indices:
                    if used_unfull_num < unfull_num:
                        if indices[0] != 0:
                            indices[0] = indices[0] + 1 + used_unfull_num
                        indices[1] = indices[1] + 1 + used_unfull_num
                        used_unfull_num += 1
            else:
                slice_indices[-1][1] += unfull_num



    def _optimization(self, data, common_comp, map_matrices,
                      z_loss = 1e-4, z_count = 200, p_loss = 1e-4, p_count = 200, total_loss = 5e-7, total_count = 200):
        # iterative update reduced_data and map_matrix
        all_coveraged = None
        iteration = 0
        while True:
            # update all common component of each sample vector
            for i in range(self.num_sample):
                weight_map_i = np.sum(self.hotkernel_mat[i, :])
                weighted_comp_i = np.dot(self.hotkernel_mat[i, :].reshape(1, -1), common_comp)

                # optimize the component based on map matrix
                optimized_count = 0
                coveraged = None
                # optimize common component zi
                while True:
                    r = []
                    Q = []
                    # calculate the r of objective function
                    for v_id in range(self.num_view):
                        mapped_data = np.dot(map_matrices[v_id], data[v_id][i, :].T)
                        r.append(common_comp[i, :] - mapped_data)

                    # calculate the weight function Q in the objective function
                    for v in range(self.num_view):
                        Q.append(1 / (np.dot(r[v], r[v].T) + self.alpha ** 2))
                    sum_Q = sum(Q)

                    updated_common_comp_i = None
                    # assemble the two part of the objective function a and b(object = a^-1 * b)
                    if self.algorithm == "LPP":
                        # the assemble of 'a' should be
                        a = self.num_view * self.lambda2 \
                            + self.num_view * self.lambda3 * weight_map_i \
                            + sum_Q

                        # the assemble of 'b' should be
                        b1 = self.num_view * self.lambda2 * self.onehots[0][i, :].reshape(1,-1)
                        b2 = self.num_view * self.lambda3 * weighted_comp_i
                        b = b1 + b2
                        for v in range(self.num_view):
                            b = b + Q[v] * np.dot(map_matrices[v], data[v][i, :])
                        updated_common_comp_i = np.linalg.inv(np.eye(self.subspace_dim) * a).dot(b.reshape(-1, 1))

                    if self.algorithm == "LPDP":
                        # the LPDP require an extra adjacency matrix to describe Sw and Sb(replace as St - Sw)
                        sum_j_Mi = np.sum((self.M[i,:])) * 2

                        # the assemble of 'a' should be
                        a = self.num_view * self.lambda2 \
                            + self.num_view * self.lambda3 * weight_map_i \
                            - sum_j_Mi * self.lambda4 \
                            + sum_Q

                        # the assemble of 'b' should be
                        temp_indices = np.arange(self.num_sample)
                        temp_indices = temp_indices.tolist()
                        temp_indices.pop(i)
                        sum_comps_without_compi = common_comp[temp_indices].sum(axis = 0)
                        # b = self.num_view * self.lambda2 * onehots[0][i, :] \
                        #     + self.num_view * self.lambda3 * weighted_comp_i \
                        #     - self.lambda4 * sum_comps_without_compi * 2
                        b = self.num_view * self.lambda2 * self.onehots[0][i, :] \
                            + self.num_view * self.lambda3 * weighted_comp_i \
                            - self.lambda4 * sum_comps_without_compi * 2
                        for v in range(self.num_view):
                            b = b + Q[v] * np.dot(map_matrices[v], data[v][i, :])
                        updated_common_comp_i = np.linalg.inv(np.eye(self.subspace_dim) * a).dot(b.reshape(-1, 1))

                    # estimate loss of this component and break the loop if coverage
                    loss = updated_common_comp_i.reshape(1, -1) - common_comp[i, :]
                    loss = np.dot(loss, loss.T)
                    loss = np.sqrt(loss)
                    # update
                    common_comp[i, :] = updated_common_comp_i.reshape(1,-1)
                    optimized_count += 1
                    if loss < z_loss:
                        coveraged = True
                        break
                    if optimized_count > z_count:
                        coveraged = False
                        break

            # optimize map matrix based on common_component
            optimized_count = 0
            coveraged = None

            # update map matrices in view width
            last_map_matrices = tuple(map_matrices)
            for v in range(self.num_view):
                map_mat_v = map_matrices[v]
                # optimize mapping matrix on view v
                while True:
                    G = []
                    r = []
                    # calculate the G of objective function
                    for sample_id in range(self.num_sample):
                        mapped_data = np.dot(map_mat_v, data[v][sample_id, :].T)
                        r_i = common_comp[sample_id, :] - mapped_data
                        r.append(r_i)
                        G.append(1 / (np.dot(r_i, r_i.T) + self.alpha ** 2))
                    X = data[v]
                    G = np.array(G).reshape(-1, 1)
                    # XG = np.dot(X.T, G * np.eye(self.num_sample)).T
                    # XGX = np.dot(X.T, XG)
                    XGX, XGZ = None, None
                    for i in range(self.num_sample):
                        xi_Gi = X[i, :] * G[i]
                        xi = X[i, :]
                        if XGX is None:
                            XGX = np.outer(xi_Gi, xi)
                        else:
                            XGX = XGX + np.outer(xi_Gi, xi)
                        if XGZ is None:
                            XGZ = np.outer(xi_Gi, common_comp[i, :])
                        else:
                            XGZ = XGZ + np.outer(xi_Gi, common_comp[i, :])

                    # assume the structure of objective function should be like a^-1 * b,
                    # the assemble of 'a' should be
                    a = XGX + self.lambda1 * self.num_sample * np.eye(self.num_dim)
                    # the assemble of 'b' should be XGZ
                    # GZ = np.dot(G * np.eye(self.num_sample),
                    #             common_comp)
                    # b = np.dot(common_comp.T, XG)
                    b = XGZ
                    updated_map_mat_v = np.dot(np.linalg.inv(a), b).T
                    # use Frobenius norm to estimate loss of mapping matrix
                    loss = updated_map_mat_v - map_mat_v
                    loss = loss * loss
                    loss = np.sum(loss)
                    loss = np.sqrt(loss)
                    optimized_count += 1

                    # update for next loop
                    map_mat_v = updated_map_mat_v

                    if loss < p_loss:
                        coveraged = True
                        break
                    if optimized_count > p_count:
                        coveraged = False
                        break
                    pass
                # obtain optimized map vector
                map_matrices[v] = updated_map_mat_v
            iteration += 1
            # now estimate the max loss of map matrixes
            max_loss = 0
            for map_id in range(len(map_matrices)):
                loss = last_map_matrices[map_id] - map_matrices[map_id]
                loss = loss * loss
                loss = np.sqrt(np.sum(loss))
                max_loss = max(loss, max_loss)
            if max_loss < total_loss:#5e-7:
                all_coveraged = True
                break
            if iteration > total_count:
                all_coveraged = False
                break
            pass
        return map_matrices, common_comp

    def test(self, test_data, labels, map_matrices, mode, K_Near):
        # testing method aim to find each mapped data vectors nearest vector (use Eucilid distance). If the nearest
        # vector and selected vector shared with same label, the selected vector was mapped to a proper manifold.
        # TODO: design a independent cross-validation method(or refactor train, isolate the part related to num_class
        #  values, and optimized part as a independent method). The frame work need to be implement is：
        #  1: random split 2:k-fold

        # map data to manifold described by mapping matrices
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics.cluster import normalized_mutual_info_score
        num_view = self._obtain_numview(test_data)
        num_sample = test_data[0].shape[0]
        mapped_data = []
        for vi in range(num_view):
            mapped_vi = np.dot(test_data[vi], map_matrices[vi].T)
            mapped_data.append(mapped_vi)
            labels[vi] = np.ravel(labels[vi])

        # count the accuracy of data in the manifold
        acc_list = []
        NMI_list = []
        for vi in range(num_view):
            neigh = KNeighborsClassifier(n_neighbors=K_Near)
            _ =neigh.fit(mapped_data[vi], labels[vi])
            for vj in range(num_view):
                if mode == "pair_wise":
                    if vi != vj:
                        pred = neigh.predict(mapped_data[vj])
                        label = labels[vj]
                        label = np.ravel(label)
                        t = pred - label
                        count = 0
                        for ele_idx in range(max(t.shape)):
                            if t[ele_idx] == 0:
                                count += 1
                        acc = count / num_sample
                        acc_list.append(acc)
                        NMI_score = normalized_mutual_info_score(pred, label)
                        NMI_list.append(NMI_score)

        acc_ave = sum(acc_list) / len(acc_list) * 100
        NMI_ave = sum(NMI_list) / len(NMI_list) * 100
        D_acc = []
        for acc in acc_list:
            D_acc.append((acc-acc_ave)**2)
        D_acc_ave = sum(D_acc) / len(D_acc)
        return acc_ave, NMI_ave, D_acc_ave

    def _get_hotkernel_mat(self, label, train_data):
        # obtain similarity matrix on multi_view
        Sv = None
        for view_label, view_data in zip(label, train_data):
            # for each view
            similarity_mat = np.zeros((self.num_sample, self.num_sample))
            for i in range(self.num_sample):
                # for each pair of vectors in data
                vector_i = view_data[i, :]
                for j in range(self.num_sample):
                    vector_j = view_data[j, :]
                    e_v = vector_i - vector_j
                    similarity_mat[i, j] = np.exp(-np.dot(e_v, e_v.T) / self.sigma)
            if Sv is None:
                Sv = similarity_mat
            else:
                Sv += similarity_mat
        Sv = Sv / self.num_view
        # obtain hot kernel matrix on the basis of similarity matrix and respective data labels
        for i in range(self.num_sample):
            for j in range(self.num_sample):
                if label[0][0, i] != label[0][0, j]:  # only consider the similarity between data shared with same labels
                    Sv[i, j] = 0
        return Sv

    def _num2onehot(self, labels):
        # 以(num_sample, num_classes)的格式组织独热编码，假设训练数据都包含了所有的类
        onehots = []
        for view in labels:
            onehot = np.zeros((max(view.shape), self.num_class))
            for dim1 in range(max(view.shape)):
                l = view[0, dim1] - self.min_class
                onehot[dim1, l] = 1
            onehots.append(onehot)
        return onehots

    def _obtain_numclass(self, label_set):
        # obtain number of class in the label_set.
        # assume all classes have been present in the label_set
        # assume all the labels from each view was aligned based on same data object it belongs to
        # obtain max label and min label among all dataset
        min_l, max_l = np.min(label_set[0]), \
                       np.max(label_set[0])
        num_classes = max_l - min_l + 1
        # annotation this code,
        # To do: add a method  to recompile label when input labels into train method to make sure labels are distinct
        # # make sure the num_classes is about the number of distinct label among the label set
        # bitmap = np.ones(num_classes)
        # for l_id in range(max(label_set[0].shape)):
        #     bitmap[label_set[0][0,l_id]-min_l] = 0
        # idices = bitmap[bitmap == 1].tolist()
        # num_classes = np.sum(bitmap)
        return num_classes, max_l, min_l

    def _obtain_numsample(self, data):
        # obtain number of samples in the label vectors
        return data.shape[0]

    def _obtain_numdim(self, data):
        # obtain dimension of data
        return data.shape[1]

    def _obtain_numview(self, data_set):
        # obtain number of view the dataset have
        return len(data_set)

K_Near = 1

if __name__ == "__main__":
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
    # param = io.loadmat("E:\\Works\\MatlabWork\\MvCCDA_mnist_usps\\param.mat")
    param = dataloader.load_param()
    common_comp = param.get('Z').T
    map_matrices = param.get('P')
    for map_id in range(len(map_matrices)):
        map_matrices[map_id] = map_matrices[map_id].T
    # map_matrices = pca_loader.cell2list(map_matrices)
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
    # algorithm="LPP", sigma = 200, lambda1=0.6, lambda2=0.02, lambda3=5e-2 # /k=1 5s 82.0% 6590 73% / k=5 21s 81.2% 6462 73% / k=7 21s 81.4% 6494 73%
    # algorithm="LPDP", t = 1, sigma = 200, lambda1=0.6, lambda2=0.2, lambda3=0, lambda4=5e-4 # / k=1 7-8s 82.2% 6622 74% / k=3 7s 82.2% 6622 74.8% / k=5 7s 82.6% 6686 75% / k=7 7s 83.8% 6882 76.4%
    # algorithm="MMMC", t = 1, sigma = 200, lambda1=0.06, lambda2=0.2, lambda3=0, lambda4=5e-4 # / k=1 8s 82% 6590 72% / k=3 7s 82.6% 6687 73% / k=5 8s 83.8% 6883 75% / k=7 84.4% 6981 75%
    #
    #

    # pie                                                                           # time acc nmi
    # algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-5 # 19s 72.3% 89.6%
    # algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-4 # 19s 72.3% 89.6%
    # algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=5e-4 # /k=1 70s 79%  91% /k=3 77%  90% /k=5 70s 76%  89% /k=7 69s 73.9%  88%
    #
    # algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-5, lambda4=5e-5 # 20s 72.44% 89.6%
    # algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-5 # 20s 72.44% 89.6%
    # algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=5e-6, lambda4=5e-5 # 87s 79% 91%
    # algorithm="LPDP", t = 2, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-3, lambda4=5e-3 # /k=1 85s 79.4% 92% /k=3 78.42% /k=5 76.57% /k=7 75%
    #

    # algorithm="MMC", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 60s 77.99% 91.4%
    # algorithm="MMMC", t = 0.1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 41s 75.33% 90.5%
    # algorithm="MMMC", t = 0.01, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 40s 74.67% 90.4%
    # algorithm="MMMC", t = 1.5, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 75s 78.75% 91.5%
    # algorithm="MMMC", t = 2, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # /k=1 87s 79.29% 91.8% /k=3 78.09% /k=5 76.35% /k=7 74%
    #
    # algorithm="LPP", t = 1 ,sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=0, lambda4=0 # /k=1 75s 79% 91%/k=3 77.88 /k=5 76.03% /k=7 73%
    # algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0 # 20s 72.34% 89.6%
    # algorithm="LPP", t = 1, sigma = 2000, lambda1=0.06, lambda2=0.02, lambda3=0 # 22s 49.56% 80.93%
    #
    import time
    model = MvCCDA(
        algorithm="LPP", t = 1 ,sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=0, lambda4=0 # 75s 79% 91%
    )

    t1 = time.time()
    map_matrices, common_comp = model.train(tr_MvData, tr_label, rand_seed=0,)
    t2 = time.time()
    t = t2 - t1
    print("training done")

    # ave_acc, ave_nmi, ave_D_acc = model.test(te_MvData, te_label, map_matrices, mode="pair_wise", K_Near=1)
    # print("t: " + str(model.t) + ", l1: " +str(model.lambda1) +", l2: "+str(model.lambda2)+", l3: "+str(model.lambda3)+", l4: "+str(model.lambda4))
    # print("ave_acc: " + str(ave_acc) + " ave_nmi: " + str(ave_nmi))
    # print("time: " + str(t))

    for i in range(1,9,2):
        ave_acc, ave_nmi, ave_D_acc = model.test(te_MvData, te_label, map_matrices, mode ="pair_wise", K_Near=i)
        print(i)
        print("time: "+str(t))
        print("ave_acc: "+str(ave_acc) + " ave_nmi: " + str(ave_nmi))
    #
    # print("percentage of how many vector and it's nearest vector shared with same label among unmapped dataset: \n"+str(unmapped_ave_acc)+"%")
    # print("percentage of how many vector and it's nearest vector shared with same label among mapped dataset: \n" + str(mapped_ave_acc)+"%")
    pass
