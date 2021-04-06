import numpy as np
from Data import dataloader
from scipy.linalg import eigh


class MvCCDA():
    def __init__(self, alpha = 2.3849, lambda1 = 0.6,lambda2 = 0.002,lambda3 = 0.0005,sigma = 2000,subspace_dim = 45):

        # initial parameters of hyper setting
        # hyperparameters = {"alpha": alpha, "lambda1": lambda1, "lambda2": lambda2, "lambda3": lambda3, "sigma": sigma,
        #                    "subspace_dim": subspace_dim}
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.sigma = sigma
        self.subspace_dim = subspace_dim


        # self.data_param = {"num_sample": num_sample, "num_dim": num_dim, "num_view": num_view}


    def train(self, train_data,
              common_comp:np.array = None, map_matrices:list = None, rand_seed = None):
        # initial parameters about input data
        self.num_sample = self._obtain_numsample(train_data[0])
        self.num_dim = self._obtain_numdim(train_data[0])
        self.num_view = self._obtain_numview(train_data)
        # initial matrices which will be used in the update procedures
        onehots, _ = self._num2onehot(labels)
        hotkernel_mat = self._get_hotkernel_mat(labels, train_data)

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
                map_matrices.append(np.random.random((self.num_dim, self.subspace_dim)))
        pass

        # iterative update reduced_data and map_matrix
        all_coveraged = None
        iteration = 0
        while True:
            # update all common component of each sample vector
            for i in range(self.num_sample):
                weight_map_i = np.sum(hotkernel_mat[i, :])
                weighted_comp_i = np.dot(hotkernel_mat[i, :].reshape(1, -1), common_comp)

                # optimize the component based on map matrix
                optimized_count = 0
                coveraged = None
                # optimize common component zi
                while True:
                    r = []
                    Q = []
                    # calculate the r of objective function
                    for v_id in range(self.num_view):
                        mapped_data = np.dot(map_matrices[v_id], train_data[v_id][i, :].T)
                        r.append(common_comp[i, :] - mapped_data)

                    # calculate the weight function Q in the objective function
                    for v in range(self.num_view):
                        Q.append(1 / (np.dot(r[v], r[v].T) + self.alpha ** 2))
                    sum_Q = sum(Q)

                    # assemble the two part of the objective function
                    a = self.num_view * self.lambda2 \
                        + self.num_view * self.lambda3 * weight_map_i \
                        + sum_Q

                    b = self.num_view * self.lambda2 * onehots.get("train")[0][i,
                                                                :] + self.num_view * self.lambda3 * weighted_comp_i
                    for v in range(self.num_view):
                        b = b + Q[v] * np.dot(map_matrices[v], train_data[v][i, :])

                    # estimate loss of this component and break the loop if coverage
                    updated_common_comp_i = np.linalg.inv(np.eye(self.subspace_dim) * a).dot(
                        b.reshape(-1, 1))
                    loss = updated_common_comp_i.reshape(1, -1) - common_comp[i, :]
                    loss = np.dot(loss, loss.T)
                    loss = np.sqrt(loss)
                    # update
                    common_comp[i, :] = updated_common_comp_i.T
                    optimized_count += 1
                    if loss < 0.0001:
                        coveraged = True
                        break
                    if optimized_count > 200:
                        coveraged = False
                        break
            # optimize map matrix based on common_component
            optimized_count = 0
            coveraged = None

            # update component in view width
            last_map_matrixes = tuple(map_matrices)
            for v in range(self.num_view):
                map_mat_v = map_matrices[v]
                # optimize mapping matrix on view v
                while True:
                    G = []
                    r = []
                    # calculate the G of objective function
                    for sample_id in range(self.num_sample):
                        mapped_data = np.dot(map_mat_v, train_data[v][sample_id, :].T)
                        r_i = common_comp[sample_id, :] - mapped_data
                        r.append(r_i)
                        G.append(1 / (np.dot(r_i, r_i.T) + self.alpha ** 2))
                    data_X = train_data[v]
                    G = np.array(G).reshape(-1, 1)
                    # assuming the structure of objective function should be like b / a,
                    # the assemble of 'a' should be
                    XG = np.dot(G * np.eye(self.num_sample),
                                data_X)
                    XGX = np.dot(data_X.T, XG)
                    a = XGX + self.lambda1 * self.num_sample * np.eye(
                        self.num_dim)
                    # the assemble of 'b' should be
                    # b = np.dot(common_comp.T, XG)
                    GZ = np.dot(G * np.eye(self.num_sample),
                                common_comp)
                    b = np.dot(common_comp.T, XG)
                    updated_map_mat_v = np.dot(np.linalg.inv(a), b.T).T

                    # use Frobenius norm to estimate loss of mapping matrix
                    loss = updated_map_mat_v - map_mat_v
                    loss = loss * loss
                    loss = np.sum(loss)
                    loss = np.sqrt(loss)
                    optimized_count += 1

                    # update for next loop
                    map_mat_v = updated_map_mat_v

                    if loss < 1e-4:
                        coveraged = True
                        break
                    if optimized_count > 200:
                        coveraged = False
                        break
                    pass
                # obtain optimized map vector
                map_matrices[v] = updated_map_mat_v
            iteration += 1
            # now estimate the max loss of map matrixes
            max_loss = 0
            for map_id in range(len(map_matrices)):
                loss = last_map_matrixes[map_id] - map_matrices[map_id]
                loss = loss * loss
                loss = np.sqrt(np.sum(loss))
                max_loss = max(loss, max_loss)
            if max_loss < 5e-7:
                all_coveraged = True
                break
            if iteration > 200:
                all_coveraged = False
                break
            pass
        return map_matrices, common_comp

    def test(self, test_data, labels, map_matrices):
        # testing method aim to find each mapped data vectors nearest vector (use Eucilid distance). If the nearest
        # vector and selected vector shared with same label, the selected vector was mapped to a proper manifold.

        # map data to manifold described by mapping matrices
        from sklearn.neighbors import KNeighborsClassifier
        num_view = self._obtain_numview(test_data)
        num_sample = test_data[0].shape[0]
        mapped_data = []
        for vi in range(num_view):
            mapped_vi = np.dot(test_data[vi], map_matrices[vi].T)
            mapped_data.append(mapped_vi)

        # count the accuracy of data in the manifold
        acc_list = []
        for vi in range(num_view):
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(mapped_data[vi], labels[vi].reshape(-1, 1))
            for vj in range(num_view):
                if vi != vj:
                    pred = neigh.predict(mapped_data[vj])
                    label = labels[vj]
                    t = pred - label
                    count = 0
                    for ele_idx in range(max(t.shape)):
                        if t[0, ele_idx] == 0:
                            count += 1
                    acc = count / num_sample
                    acc_list.append(acc)
        acc_ave = sum(acc_list) / len(acc_list) * 100
        return acc_ave

    def _get_hotkernel_mat(self, label_set, train_data):
        _, max_l, min_l = self._obtain_numclass(label_set)
        # obtain similarity matrix on multi_view
        Sv = None
        label = label_set.get("train")
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
                if label[0][0, i] != label[0][0, j]:  # only consider the similarity of data shared with same labels
                    Sv[i, j] = 0
        return Sv

    def _num2onehot(self, label_set):
        # 以(num_sample, num_classes)的格式组织独热编码，假设训练数据都包含了所有的类
        num_classes, max_l, min_l = self._obtain_numclass(label_set)
        v_labels = {}
        for data_type in label_set.keys():
            v_labels.update({data_type: None})
        for data_type in label_set.keys():
            labels = label_set.get(data_type)
            onehots = []
            for view in labels:
                onehot = np.zeros((view.shape[1], num_classes))
                for dim1 in range(view.shape[1]):
                    l = view[0, dim1] - min_l
                    onehot[dim1, l] = 1
                onehots.append(onehot)
            v_labels.update({data_type: onehots})
        pass
        return v_labels, num_classes

    def _obtain_numclass(self, label_set):
        # obtain number of class in the label_set. Assuming all classes have been present in the label_set
        min_l, max_l = float("inf"), 0
        # obtain max label and min label among all dataset
        for data_type in label_set.keys():
            temp_min, temp_max = np.min(label_set.get(data_type)), \
                                 np.max(label_set.get(data_type))
            if temp_max > max_l:
                max_l = temp_max
            if temp_min < min_l:
                min_l = temp_min
        num_classes = max_l - min_l + 1
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

if __name__ == "__main__":
    train_data, test_data, labels = dataloader.load_data("matlabpca")

    # initialization of parameters to be learned, size of row data should be (num_sample, num_dim)
    param = dataloader.load_param()
    common_comp = param.get("Z").T
    map_matrixes = param.get("P")
    for m_idx in range(len(map_matrixes)):
        map_matrixes[m_idx] = map_matrixes[m_idx].T

    model = MvCCDA()
    map_matrixes, common_comp = model.train(train_data, common_comp, map_matrixes)
    pass
    # map_matrixes, common_comp = train(train_data, map_matrixes, common_comp, onehots, hotkernel_mat, train_data_param, hyperparameters)
    # project test data to subspace
    mapped_ave_acc = model.test(test_data, labels.get("test"), map_matrixes)
    #
    # print("percentage of how many vector and it's nearest vector shared with same label among unmapped dataset: \n"+str(unmapped_ave_acc)+"%")
    # print("percentage of how many vector and it's nearest vector shared with same label among mapped dataset: \n" + str(mapped_ave_acc)+"%")
    pass